import sys
import heapq
import threading
import random
import time
from redis_utils import connect_redis_list, transact_block, transact_single, execute_block, execute_single
from utils import timed
from collections import defaultdict
from document import Vocabulary


class RedisLDAModelCache:
    """
    Holds the current assumed global state and the current local deltas 
    to the LDA model.

    Currently this holds the entire local model state and can do the sync.
    """
    def __init__(self, options):
        self.rs = connect_redis_list(options.redis_hosts, options.redis_db)

        self.topics = options.topics

        # Store some metadata
        for r in self.rs:
            r.set('model', 'lda')
            r.set('topics', options.topics)
            r.set('alpha', options.alpha)
            r.set('beta', options.beta)
            r.set('document', options.document)
            r.incr('shards', 1)

        # Track the local model state
        self.topic_d = defaultdict(lambda: defaultdict(int))
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(float)
        self.topic_lock = threading.RLock()

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

        # Stat counters
        self.pulls = 0
        self.pushes = 0

        self.resample_count = 0

        # Document state
        self.documents = []
        self.v = Vocabulary()
        self.finished_loading_docs = False

        # Start threads for pull and push
        self.pull_thread_ref = threading.Thread(name="pull_thread", target=self.pull_thread)
        # self.push_thread_ref = threading.Thread(name="push_thread", target=self.push_thread)
        self.pull_thread_ref.daemon = True
        #self.push_thread_ref.daemon = True

        self.pull_thread_ref.start()
        # self.push_thread_ref.start()

    def redis_of(self, thing):
        return hash(thing) % len(self.rs)

    def push_thread(self):
        while True:
            time.sleep(120*random.random())
            if self.finished_loading_docs:
                self.push_local_state()

    def pull_thread(self):
        while True:
            time.sleep(120*random.random()) # Python doesn't have a yield
            if self.finished_loading_docs:
                self.pull_global_state()

    def lock_reset_delta_state(self):
        """
        Copy the current delta state over to a local variable and then reset it (atomic)

        """
        local_delta_topic_d = defaultdict(lambda: defaultdict(int))
        local_delta_topic_w = defaultdict(lambda: defaultdict(int))
        local_delta_topic_wsum = defaultdict(int)

        # Lock and make a copy of the current delta state
        with self.topic_lock:
            # Normally I would use deepcopy for this, but it barfs inside of cython... :-/
            for z,v in self.delta_topic_d.iteritems():
                for d, delta in v.iteritems():
                    if delta != 0:
                        local_delta_topic_d[z][d] = delta
            for z,v in self.delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    if delta != 0:
                        local_delta_topic_w[z][w] = delta
            for z, delta in self.delta_topic_wsum.iteritems():
                if delta != 0:
                    local_delta_topic_wsum[z] = delta

            # Reset the deltas
            self.delta_topic_d = defaultdict(lambda: defaultdict(int))
            self.delta_topic_w = defaultdict(lambda: defaultdict(int))
            self.delta_topic_wsum = defaultdict(int)

        return local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum


    @timed("push_local_state")
    def push_local_state(self):
        """
        Push our current set of deltas to the server
        """
        local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum = self.lock_reset_delta_state()

        # This a potentially long, blocking IO operation
        # sys.stderr.write('Push local state...\n')
        with execute_block(self.rs, transaction=False) as pipes:
            # Update document state from deltas
            for z,v in local_delta_topic_d.iteritems():
                for d, delta in v.iteritems():
                    pipes[self.redis_of(d)].zincrby(('d', z), d, delta)
                pipes[self.redis_of(d)].zremrangebyscore(('d', z), 0, 0)

            # Update topic state
            for z,v in local_delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    if delta != 0:
                        pipes[self.redis_of(w)].zincrby(('w', z), w, delta)
            # Update sums
            for z, delta in local_delta_topic_wsum.iteritems():
                if delta != 0:
                    pipes[self.redis_of(z)].hincrby('wsum', z, delta)

            # Prune zeros from the w hash keys to save memory
            for pipe in pipes:
                for z in range(self.topics):
                    pipe.zremrangebyscore(('w', z), 0, 0)

        self.pushes += 1


    @timed("pull_global_state")
    def pull_global_state(self):
        # Note we don't need to pull the d state, since our shard is 100% responsible for it

        # Push the local state first; not sure of the impact of this, thought it roughly doubles the number of push synchronizations
        # The delta re-apply logic below will actually take care of the inconsistencies this introduces
        self.push_local_state()

        # TODO: this logic leads to large transient memory spikes in each redis shard (say 3-4x baseline mem usage), when
        #       client shards pile on. I haven't narrowed it down to whether it is the zrevrangebyscore or hgetall
        #       but neither seem to be particularly likely candidates; 

        local_topic_w = defaultdict(lambda: defaultdict(int))
        local_topic_wsum = defaultdict(int)

        # TODO: this part can be pipelined as well
        # Pull down the global state (wsum and topic_w should be in the same transaction)
        with transact_block(self.rs) as pipes:
            # Remove everything with zero count to save memory
            for pipe in pipes:
                for z in range(self.topics):
                    # This is clever because it avoids sending the zeros over the wire
                    pipe.zrevrangebyscore(('w', z), float('inf'), 1, withscores=True)
                for z, zz in enumerate(pipe.execute()):
                    for w,v in zz:
                        local_topic_w[z][int(w)] = int(v)

                for z,c in pipe.hgetall('wsum').execute()[0].iteritems():
                    local_topic_wsum[int(z)] = int(c)

        # Inform the running model about the new state
        # But, by this point we may actually be behind our own local state, so lock the deltas and apply them again
        # (this is the price we pay for not locking over the entire pull)
        with self.topic_lock:
            self.topic_w = local_topic_w
            self.topic_wsum = local_topic_wsum

            # Add in the current deltas
            for z,v in self.delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    if delta != 0:
                        self.topic_w[z][w] += delta
            for z, delta in self.delta_topic_wsum.iteritems():
                if delta != 0:
                    self.topic_wsum[z] += delta

        self.pulls += 1

    
    def topic_to_string(self, topic, max_length=20):
        result = []
        for w,c in topic.iteritems():
            if len(result) > max_length:
                heapq.heappushpop(result, (c,self.v.rev(w)))
            else:
                heapq.heappush(result, (c,self.v.rev(w)))
        return heapq.nlargest(max_length, result)


    def add_d_w(self, d, w, z=None):
        """
        Add word w to document d
        """
        d.assignment.append(z)

        with self.topic_lock:
            self.topic_d[z][d.id] += 1
            self.topic_w[z][w] += 1
            self.topic_wsum[z] += 1

            self.delta_topic_d[z][d.id] += 1
            self.delta_topic_w[z][w] += 1
            self.delta_topic_wsum[z] += 1

    def move_d_w(self, w, d, i, oldz, newz):
        """
        Move w from oldz to newz
        """
        if newz != oldz:
            with self.topic_lock:
                self.topic_d[oldz][d.id] += -1
                self.topic_w[oldz][w] += -1
                self.topic_wsum[oldz] += -1
                assert self.topic_d[oldz][d.id] >= 0
                assert self.topic_w[oldz][w] >= 0
                assert self.topic_wsum[oldz] >= 0

                self.topic_d[newz][d.id] += 1
                self.topic_w[newz][w] += 1
                self.topic_wsum[newz] += 1

                self.delta_topic_d[oldz][d.id] += -1
                self.delta_topic_w[oldz][w] += -1
                self.delta_topic_wsum[oldz] += -1

                self.delta_topic_d[newz][d.id] += 1
                self.delta_topic_w[newz][w] += 1
                self.delta_topic_wsum[newz] += 1

            d.assignment[i] = newz

    def finalize_iteration(self, iter):
        if iter == 0:
            with execute_single(self.rs[0], transaction=False) as pipe:
                for w,id in self.v.to_id.iteritems():
                    pipe.hset('lookup', id, w)
        for r in self.rs:
            r.incr('iterations')


def dump_model(rs):
    import gzip
    import json

    d = {
        'model':      rs[0].get('model'),
        'document':   rs[0].get('document'),
        'shards':     int(rs[0].get('shards')),
        'iterations': int(rs[0].get('iterations')),
        'topics':     int(rs[0].get('topics')),
        'alpha':      float(rs[0].get('alpha')),
        'beta':       float(rs[0].get('beta')),
        'lookup':     rs[0].hgetall('lookup'),
        'w':          defaultdict(lambda: defaultdict(int)),
        'd':          defaultdict(lambda: defaultdict(int))
        }

    doc_name = d['document'].split('/')[-1]

    with gzip.open('MODEL_%s-%s-T%d-alpha%.3f-beta%.3f-effective_iter=%d.json.gz' % (d['model'], doc_name, d['topics'], d['alpha'], d['beta'], int(d['iterations'] / float(d['shards']))), 'w') as f:
        with transact_block(rs) as pipes:
            for pipe in pipes:
                for z in range(d['topics']):
                    # This is clever because it avoids sending the zeros over the wire
                    pipe.zrevrangebyscore(('w', z), float('inf'), 1, withscores=True)
                # get words
                for z, zz in enumerate(pipe.execute()):
                    for w,v in zz:
                        v = int(v)
                        assert v > 0
                        d['w'][z][w] = v
                # get documents
                for z in range(d['topics']):
                    pipe.zrevrangebyscore(('d', z), float('inf'), 1, withscores=True)
                for z, zz in enumerate(pipe.execute()):
                    for doc,v in zz:
                        d['d'][z][doc] = int(v)


        f.write(json.dumps(d))

