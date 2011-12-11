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

# NOTE: waffled on having monolithic or fine-grained keys; monolithic causes too much blocking
# TODO: invert z->w to w->z and push one per word

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
        self.pulls = 0
        self.pushes = 0

        self.resample_count = 0

    def post_initialize(self):
        """
        Start a thread for synchronizing state with the redis
        """
        self.pull_thread_ref = threading.Thread(name="pull_thread", target=self.pull_thread)
        self.pull_thread_ref.daemon = True
        self.pull_thread_ref.start()

    def pull_thread(self):
        while True:
            self.sync_state()
            # TODO: make this a function of iterations not time
            # time.sleep(1200*random.random()) # Python doesn't have a yield
            time.sleep(0.1) # Python doesn't have a yield

    def redis_of(self, thing):
        return hash(thing) % len(self.rs)

    @timed("lock_fork_delta_state")
    def lock_fork_delta_state(self):
        """
        Copy the current delta state over to a local variable and then reset it (atomic)
        """
        # Lock and make a copy of the current delta state
        with self.topic_lock:
            # Normally I would use deepcopy for this, but it barfs inside of cython... :-/
            local_delta_topic_d = copy_sparse_defaultdict_2(self.delta_topic_d)
            local_delta_topic_w = copy_sparse_defaultdict_2(self.delta_topic_w)
            local_delta_topic_wsum = copy_sparse_defaultdict_1(self.delta_topic_wsum)

            # Reset the deltas
            self.delta_topic_d = defaultdict(lambda: defaultdict(int))
            self.delta_topic_w = defaultdict(lambda: defaultdict(int))
            self.delta_topic_wsum = defaultdict(int)

        return local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum

    def lock_merge_delta_state(self, local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum):
        with self.topic_lock:
            merge_defaultdict_2(self.delta_topic_d, local_delta_topic_d)
            merge_defaultdict_2(self.delta_topic_w, local_delta_topic_w)
            merge_defaultdict_1(self.delta_topic_wsum, local_delta_topic_wsum)

    @with_retries(10, "push_local_state_redis")
    def push_local_state_redis(self, local_delta_topic_d, local_delta_topic_w):
        """
        Push local state deltas transactionally (important to maintain state / bc other consumers might make pull requests)
        """
        # push d state (back to one key per document); this does not need to be in the same block as w since
        # only the model listener process is looking at this information
        with timing("pushing d state"):
            with execute_block(self.rs, transaction=True) as pipes:
                # Update document state from deltas
                for d,v in local_delta_topic_d.iteritems():
                    for z, delta in v.iteritems():
                        pipes[self.redis_of(d)].zincrby(to_key('d',d), z, delta)
            with execute_block(self.rs, transaction=True) as pipes:
                for doc in self.documents:
                    d = doc.id
                    pipes[self.redis_of(d)].zremrangebyscore(to_key('d',d), 0, 0)






        # XXXX: what happens on partial push / fail here; these need to be the same transaction




        # w and d don't need to be in the same block ; have one key per word for even more fine-grained parallelism
        with timing("pushing w state"):
            with execute_block(self.rs, transaction=True) as pipes:
                # Update topic state
                for w,v in local_delta_topic_w.iteritems():
                    for z, delta in v.iteritems():
                        if delta != 0:
                            pipes[self.redis_of(w)].zincrby(to_key('w',w), z, delta)

        # Prune zeros from the w hash keys to save memory (opportunisitically non-transactionally)
        if random.random() < 1.0 / self.options.shards:
            with timing("pushing zremrange w"):
                with execute_block(self.rs) as pipes:
                    for w in iter(local_delta_topic_w):
                        pipes[self.redis_of(w)].zremrangebyscore(to_key('w',w), 0, 0)

    @with_retries(10, "pull_global_state_redis")
    def pull_global_state_redis(self):
        # TODO: this logic leads to large transient memory spikes in each redis shard (say 3-4x baseline mem usage), when
        #       client shards pile on. I haven't narrowed it down to whether it is the zrevrangebyscore or hgetall
        #       but neither seem to be particularly likely candidates; 

        local_topic_w = defaultdict(lambda: defaultdict(int))
        local_topic_wsum = defaultdict(int)

        # Pull down the global state 
        assert len(self.rs) == 1  # sorry for now only support single redii
        with timing("pulling w state"):
            # XXX: don't think this needs to be transactional because each w/zrevrange command is atomic
            # XXX: pipes[0] here too
            for w in iter(self.v.to_word):
                print w
                zvs = self.rs[self.redis_of(w)].zrevrangebyscore(to_key('w',w), float('inf'), 1, withscores=True)
                for (z,v) in zvs:
                    local_topic_w[int(w)][int(z)] = int(v)
                    local_topic_wsum[int(z)] += int(v)

        return local_topic_w, local_topic_wsum

    @timed("sync_state")
    def sync_state(self):
        # Note we don't need to pull the d state, since our shard is 100% responsible for it

        # Push the local state first; not sure of the impact of this, thought it roughly doubles the number of push synchronizations
        # The delta re-apply logic below will actually take care of the inconsistencies this introduces
        local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum = self.lock_fork_delta_state()

        # Man, wouldn't it be awesome if python had more expressive callbacks?
        (success, _) = self.push_local_state_redis(local_delta_topic_d, local_delta_topic_w)

        if success:
            self.pushes += 1
        else:
            # on failure, merge our forked deltas back with the trunk; we'll retry later
            sys.stderr.write('ERROR: push failed after 10 retries; merging local deltas\n')
            self.lock_merge_delta_state(local_delta_topic_d, local_delta_topic_w, local_delta_topic_wsum)

        # Pull the global state
        (success, (local_topic_w, local_topic_wsum)) = self.pull_global_state_redis()

        if success:
            # Inform the running model about the new state
            # But, by this point we may actually be behind our own local state, so lock the deltas and apply them again
            # (this is the price we pay for not locking over the entire pull)
            with self.topic_lock:
                # self.topic_w now has reference to local_topic_w)
                self.topic_w = merge_defaultdict_2(local_topic_w, self.delta_topic_w)
                self.topic_wsum = merge_defaultdict_1(local_topic_wsum, self.delta_topic_wsum)

            self.pulls += 1

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
            w_keys = pipes[0].keys(to_key('w','*'))

            for w_key in w_keys:
                pipes[0].zrevrangebyscore(w_key, float('inf'), 1, withscores=True)
            _, w = from_key(w_key)
            for w, zvs in pipes[0].execute():  # pipe[0], this is why we assert above
                for (z,v) in zvs:
                    d['w'][int(z)][int(w)] = int(v)
            d_keys = pipes[0].keys(to_key('d','*'))
            for d_key in d_keys:
                pipes[0].zrevrangebyscore(d_key, float('inf'), 1, withscores=True)
            _, d = from_key(d_key)
            for d, zvs in pipes[0].execute():  # pipe[0], this is why we assert above
                for (z,v) in zvs:
                    d['d'][int(z)][int(d)] = int(v)


        f.write(json.dumps(d))

