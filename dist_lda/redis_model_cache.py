import redis
from collections import defaultdict

class RedisLDAModelCache:
    """
    Holds the current assumed global state and the current local deltas 
    to the LDA model.

    Currently this holds the entire local model state and can do the sync.
    """
    def __init__(self, options):
        self.r = connect_redis_string(options.redis, options.redis_db)

        self.topics = options.topics

        # Store some metadata
        self.r.set('model', 'lda')
        self.r.set('topics', options.topics)
        self.r.set('alpha', options.alpha)
        self.r.set('beta', options.beta)
        self.r.set('document', options.document)
        self.r.incrby('shards', 1)

        # Track the local model state
        self.topic_d = defaultdict(lambda: defaultdict(int))
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(float)

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

        self.documents = []

        self.resample_count = 0

    @timed
    def push_local_state(self):
        """
        Push our current set of deltas to the server
        """
        # sys.stderr.write('Push local state...\n')
        with execute(self.r) as pipe:
            # Update document state from deltas
            for z,v in self.delta_topic_d.iteritems():
                for d, delta in v.iteritems():
                    if self.topic_d[z][d] == 0:  # This works because we're document sharding
                        pipe.zrem(('d', z), d)
                    else:
                        pipe.zincrby(('d', z), d, delta)

            # Update topic state
            for z,v in self.delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    if delta != 0:
                        pipe.zincrby(('w', z), w, delta)
            # Update sums
            for z, delta in self.delta_topic_wsum.iteritems():
                if delta != 0:
                    pipe.hincrby('wsum', z, delta)


        # Reset the deltas
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

    @timed
    def pull_global_state(self):
        # Note we don't need to pull the d state, since our shard is 100% responsible for it

        # XXX: always push the local state first, otherwise we'll end up with inconsistencies
        self.push_local_state()

        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(int)

        # Pull down the global state
        with transact(self.r) as pipe:
            # test using remrangebyscore(0,0) and then revrangebyscore(-inf, inf)
            for z in range(self.topics):
                pipe.zremrangebyscore(('w', z), 0, 0).execute()
            # Also count that we're not losing any data
            for z in range(self.topics):
                # This is clever because it avoids sending the zeros over the wire
                pipe.zrevrangebyscore(('w', z), float('inf'), 1, withscores=True)
            for z, zz in enumerate(pipe.execute()):
                self.topic_w[z] = defaultdict(int)
                for w,v in zz:
                    v = int(v)
                    assert v > 0
                    self.topic_w[z][w] = v

            self.topic_wsum = defaultdict(int, {int(k):int(v) for k,v in pipe.hgetall('wsum').execute()[0].iteritems()})

    
    @staticmethod
    def topic_to_string(topic, max_length=20):
        result = []
        for w,c in topic.iteritems():
            if len(result) > max_length:
                heapq.heappushpop(result, (c,w))
            else:
                heapq.heappush(result, (c,w))
        return heapq.nlargest(max_length, result)

    """
    # @timed
    def get_d(self, d):
        assert False
        topic_d = defaultdict(float)
        for z,v in self.r.hgetall(('d', d.name)).iteritems():
            topic_d[z] = float(v)
        return topic_d
    """

    def add_d_w(self, d, w, z=None):
        """
        Add word w to document d
        """
        d.assignment.append(z)

        self.topic_d[z][intern(d.name)] += 1
        self.topic_w[z][intern(w)] += 1
        self.topic_wsum[z] += 1

        self.delta_topic_d[z][intern(d.name)] += 1
        self.delta_topic_w[z][intern(w)] += 1
        self.delta_topic_wsum[z] += 1

    def move_d_w(self, w, d, i, oldz, newz):
        """
        Move w from oldz to newz
        """
        if newz != oldz:
            self.topic_d[oldz][intern(d.name)] += -1
            self.topic_w[oldz][intern(w)] += -1
            self.topic_wsum[oldz] += -1

            self.delta_topic_d[oldz][intern(d.name)] += -1
            self.delta_topic_w[oldz][intern(w)] += -1
            self.delta_topic_wsum[oldz] += -1

            self.topic_d[newz][intern(d.name)] += 1
            self.topic_w[newz][intern(w)] += 1
            self.topic_wsum[newz] += 1

            self.delta_topic_d[newz][intern(d.name)] += 1
            self.delta_topic_w[newz][intern(w)] += 1
            self.delta_topic_wsum[newz] += 1

            d.assignment[i] = newz

    def finalize_iteration(self, iter):
        self.r.incr('iterations')

