import sys
import os
import threading
import random
import time
import json
import gzip
from itertools import izip
from redis_utils import connect_redis_list, transact_block, transact_single, execute_block, execute_single
from utils import *
from collections import defaultdict
from lda_model import LDAModelCache

# TODO: abstract out the redis cache part into a generic trait
# TODO: add the LDACache back to the LDAModel

def to_key(a, b):
    return "%s/%s" % (str(a), str(b))
def from_key(k):
    a, b = k.split('/')
    return a, int(b)

class RedisLDAModelCache(LDAModelCache):
    """
    Subclass LDAModelCache with an additional thread that syncs to a redis
    """
    def __init__(self, options):
        super(RedisLDAModelCache, self).__init__(options)

        self.rs = connect_redis_list(options.redis_hosts, options.redis_db)

        self.topic_lock = threading.RLock()

        # Store some metadata
        for r in self.rs:
            r.set('model', 'lda')
            r.set('topics', options.topics)
            r.set('alpha', options.alpha)
            r.set('beta', options.beta)
            r.set('document', options.document)
            r.incr('shards', 1)

        # Stat counters
        self.syncs = 0
        self.complete_syncs = 0
        self.total_observed_weight = 0
        self.resample_count = 0

        # Construct a seed from the shard id and time
        random.seed((options.this_shard, time.time()))

    def post_initialize(self):
        """
        Start a thread for synchronizing state with the redis
        """
        self.sync_thread_ref = threading.Thread(name="sync_thread", target=self.sync_thread)
        self.sync_thread_ref.daemon = True
        self.sync_thread_ref.start()

    def sync_thread(self):
        while True:
            self.sync_state()
            # TODO: make this a function of iterations not time
            # time.sleep(1200*random.random()) # Python doesn't have a yield
            time.sleep(0.1) # Python doesn't have a yield

    def redis_of(self, thing):
        return hash(thing) % len(self.rs)

    def sync_state(self):
        # w and d don't need to be in the same transact block ; have one key per word for even more fine-grained parallelism
        self.sync_d()
        self.sync_w()

    @timed("sync_d")
    def sync_d(self):
        assert len(self.rs) == 1  # sorry for now only support single redii
        # Note we don't need to pull the d state, since our shard is 100% responsible for it

        # push d state (back to one key per document); this does not need to be in the same block as w since
        # only the model listener process is looking at this information

        # Fork a copy of the d state
        with self.topic_lock:
            local_delta_topic_d = copy_sparse_defaultdict_2(self.delta_topic_d)
            self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        try:
            with execute_block(self.rs, transaction=True) as pipes:
                # Update document state from deltas
                for d,v in local_delta_topic_d.iteritems():
                    for z, delta in v.iteritems():
                        pipes[self.redis_of(d)].zincrby(to_key('d',d), z, delta)
                for doc in self.documents:
                    d = doc.id
                    pipes[self.redis_of(d)].zremrangebyscore(to_key('d',d), 0, 0)
        except Exception, e:
            sys.stderr.write('XXX [%s] exception on push_d\n' % e)
            # Return our forked copy if there was a problem
            with self.topic_lock:
                merge_defaultdict_2(self.delta_topic_d, local_delta_topic_d, check=False)

    @timed("sync_w")
    def sync_w(self):
        # Pull the global state
        # TODO: this logic leads to large transient memory spikes in each redis shard (say 3-4x baseline mem usage), when
        #       client shards pile on. I haven't narrowed it down to whether it is the zrevrangebyscore or hgetall
        #       but neither seem to be particularly likely candidates; 
        w_keys = self.v.to_word.keys()
        random.shuffle(w_keys)
        observed_weight = 0
        for ws in grouper(self.options.sync_chunk_size, w_keys):
            # sys.stderr.write("ws=%r\n" % (ws,))
            ws = [w for w in ws if w]
            try:
                # Fork the w state
                with self.topic_lock:
                    local_delta_topic_w = defaultdict(lambda: defaultdict(int))
                    for w in ws:
                        local_delta_topic_w[w] = copy_sparse_defaultdict_1(self.delta_topic_w[w])
                        # Reset the deltas
                        self.delta_topic_w[w] = defaultdict(int)

                # Pull w state back down
                local_topic_w = defaultdict(lambda: defaultdict(int))

                # Push the state to the redis
                # Update w topic state
                with timing("increment w (%d chunk)" % self.options.sync_chunk_size):
                    # XXX: this has to be a transaction or we'll get inconsistent counts
                    with transact_block(self.rs, transaction=True) as pipes:
                        for w in ws:
                            for z, delta in local_delta_topic_w[w].iteritems():
                                if delta != 0:
                                    # TODO: this actually returns the new value; is there some way we can use this?
                                    pipes[self.redis_of(w)].zincrby(to_key('w',w), z, delta)
                            pipes[self.redis_of(w)].zremrangebyscore(to_key('w',w), 0, 0)
                        pipes[0].execute()

                        for w in ws:
                            pipes[self.redis_of(w)].zrevrangebyscore(to_key('w',w), float('inf'), 1, withscores=True)
                        for w, zv in izip(ws, pipes[0].execute()):
                            for z,v in zv:
                                local_topic_w[int(w)][int(z)] = int(v)
            except Exception, e:
                sys.stderr.write('XXX [%s] exception on push_w\n' % e)
                # Return our forked copy if there was a problem
                with self.topic_lock:
                    for w in ws:
                        merge_defaultdict_1(self.delta_topic_w[w], local_delta_topic_w[w], check=False)
            else:
                with timing("update local w state"):
                    # Inform the running model about the new state
                    # But, by this point we may actually be behind our own local state, so lock the deltas and apply them again
                    # (this is the price we pay for not locking over the entire pull)
                    with self.topic_lock:
                        for w in ws:
                            # self.topic_w now has reference to local_topic_w)
                            self.topic_w[w] = merge_defaultdict_1(local_topic_w[w], self.delta_topic_w[w])
                        # XXX: this part is slooow// probably better way to do it is to fork local w state and compare to new local
                        # rebuild wsum
                        self.topic_wsum = defaultdict(int)
                        for w,zv in self.topic_w.iteritems():
                            for z,v in zv.iteritems():
                                self.topic_wsum[z] += self.topic_w[w][z]
                                observed_weight += self.topic_w[w][z]

                    self.syncs += 1
        self.complete_syncs += 1
        self.total_observed_weight = observed_weight

    @with_retries(10, "finalize_iteration")
    def finalize_iteration(self, iter):
        if iter == 0:
            with execute_single(self.rs[0], transaction=False) as pipe:
                for w,id in self.v.to_id.iteritems():
                    pipe.hset('lookup', id, w)
        for r in self.rs:
            r.incr('iterations')


# TODO: move this inside of the class
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

    assert len(rs) == 1

    with gzip.open('MODEL_%s-%s-T%d-alpha%.3f-beta%.3f-effective_iter=%d.json.gz' % (d['model'], doc_name, d['topics'], d['alpha'], d['beta'], int(d['iterations'] / float(d['shards']))), 'w') as f:
        with transact_block(rs) as pipes:
            w_keys = pipes[0].keys(to_key('w','*')).execute()[0]
            sys.stderr.write('w_keys=%d\n' % len(w_keys))

            for w_key in w_keys:
                pipes[0].zrevrangebyscore(w_key, float('inf'), 1, withscores=True)
            for w_key, zvs in izip(w_keys, pipes[0].execute()):  # pipe[0], this is why we assert above
                _, w = from_key(w_key)
                for (z,v) in zvs:
                    d['w'][int(z)][int(w)] = int(v)
            d_keys = pipes[0].keys(to_key('d','*')).execute()[0]
            sys.stderr.write('d_keys=%d\n' % len(d_keys))
            print d_keys
            for d_key in d_keys:
                pipes[0].zrevrangebyscore(d_key, float('inf'), 1, withscores=True)
            for d_key, zvs in izip(d_keys, pipes[0].execute()):  # pipe[0], this is why we assert above
                _, doc = from_key(d_key)
                for (z,v) in zvs:
                    d['d'][int(z)][int(doc)] = int(v)


        f.write(json.dumps(d))

    
