import heapq
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

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

        self.documents = []
        self.v = Vocabulary()


        self.resample_count = 0

    def redis_of(self, thing):
        return hash(thing) % len(self.rs)

    @timed("push_local_state")
    def push_local_state(self):
        """
        Push our current set of deltas to the server
        """
        # sys.stderr.write('Push local state...\n')
        with execute_block(self.rs, transaction=False) as pipes:
            # Update document state from deltas
            for z,v in self.delta_topic_d.iteritems():
                for d, delta in v.iteritems():
                    if self.topic_d[z][d] == 0:  # This works because we're document sharding
                        pipes[self.redis_of(d)].zrem(('d', z), d)
                    else:
                        pipes[self.redis_of(d)].zincrby(('d', z), d, delta)

            # Update topic state
            for z,v in self.delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    if delta != 0:
                        pipes[self.redis_of(w)].zincrby(('w', z), w, delta)
            # Update sums
            for z, delta in self.delta_topic_wsum.iteritems():
                if delta != 0:
                    pipes[self.redis_of(z)].hincrby('wsum', z, delta)


        # Reset the deltas
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

    @timed("pull_global_state")
    def pull_global_state(self):
        # Note we don't need to pull the d state, since our shard is 100% responsible for it

        # XXX: always push the local state first, otherwise we'll end up with inconsistencies
        self.push_local_state()

        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(int)

        # Split up these two transactions for more fine-grained parallelism
        # First prune zero score hash keys
        with execute_block(self.rs, transaction=False) as pipes:
            # Remove everything with zero count to save memory
            for pipe in pipes:
                for z in range(self.topics):
                    pipe.zremrangebyscore(('w', z), 0, 0)

        # TODO: this part can be pipelined as well
        # Pull down the global state
        with transact_block(self.rs) as pipes:
            # Remove everything with zero count to save memory
            for pipe in pipes:
                for z in range(self.topics):
                    # This is clever because it avoids sending the zeros over the wire
                    pipe.zrevrangebyscore(('w', z), float('inf'), 1, withscores=True)
                for z, zz in enumerate(pipe.execute()):
                    for w,v in zz:
                        self.topic_w[z][int(w)] = int(v)

                for z,c in pipe.hgetall('wsum').execute()[0].iteritems():
                    self.topic_wsum[int(z)] = int(c)

    
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
            self.topic_d[oldz][d.id] += -1
            self.topic_w[oldz][w] += -1
            self.topic_wsum[oldz] += -1
            assert self.topic_d[oldz][d.id] >= 0
            assert self.topic_w[oldz][w] >= 0
            assert self.topic_wsum[oldz] >= 0

            self.delta_topic_d[oldz][d.id] += -1
            self.delta_topic_w[oldz][w] += -1
            self.delta_topic_wsum[oldz] += -1

            self.topic_d[newz][d.id] += 1
            self.topic_w[newz][w] += 1
            self.topic_wsum[newz] += 1

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

