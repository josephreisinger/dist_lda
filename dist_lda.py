import sys
import redis
import random
from math import log
from collections import defaultdict
from itertools import izip

from lda import *

# XXX: TODO: fix V 


class RedisLDAModelCache:
    """
    Holds the current assumed global state and the current local deltas 
    to the LDA model.

    Currently this holds the entire local model state and can do the sync.
    """
    def __init__(self, redis_instance, topics, push_every=1e5, pull_every=1e6):
        self.topics = topics
        self.r = redis_instance

        self.push_every = push_every
        self.pull_every = pull_every

        # Track the local model state
        self.topic_d = defaultdict(lambda: defaultdict(float))
        self.topic_w = defaultdict(lambda: defaultdict(float))
        self.topic_wsum = defaultdict(float)

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_w = defaultdict(lambda: defaultdict(float))
        self.delta_topic_wsum = defaultdict(float)

        self.documents = []

        self.resample_count = 0

    @timed
    def push_local_state(self):
        """
        Push our current set of deltas to the server
        """
        sys.stderr.write('Push local state...\n')
        # XXX: for now don't sync the document distributions
        # pipe.hincrby(('d', d.name), z, 1)
        with execute(self.r) as pipe:
            for z,v in self.delta_topic_w.iteritems():
                for w, delta in v.iteritems():
                    pipe.hincrby(('w', z), w, delta)
            for z, delta in self.delta_topic_wsum.iteritems():
                pipe.incr(('sum', z), amount=delta)

        # Reset the deltas
        self.delta_topic_w = defaultdict(lambda: defaultdict(float))
        self.delta_topic_wsum = defaultdict(float)

    @timed
    def pull_global_state(self):
        sys.stderr.write('Resync global model...\n')
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(float)

        with transact(self.r) as pipe:
            for z in range(self.topics):
                z = str(z)
                self.topic_w[z] = defaultdict(float)
                for w,v in pipe.hgetall(('w', z)).execute()[0].iteritems():
                    self.topic_w[z][w] = float(v)
                self.topic_wsum[z] = float(pipe.get(('sum', z)).execute()[0])

    """
    # @timed
    def get_d(self, d):
        assert False
        topic_d = defaultdict(float)
        for z,v in self.r.hgetall(('d', d.name)).iteritems():
            topic_d[z] = float(v)
        return topic_d
    """

    def check_resync(self):
        # Sync the model if necessary
        self.resample_count += 1
        if self.resample_count % self.push_every == 0:
            self.push_local_state()
        if self.resample_count % self.pull_every == 0:
            self.pull_global_state()

    def add_d_w(self, d, w, z=None):
        """
        Add word w to document d
        """
        z = str(z)
        d.assignment.append(z)

        self.topic_d[d][z] += 1
        self.topic_w[z][intern(w)] += 1
        self.topic_wsum[z] += 1

        self.delta_topic_w[z][intern(w)] += 1
        self.delta_topic_wsum[z] += 1

        self.check_resync()

    def move_d_w(self, w, d, i, oldz, newz):
        """
        Move w from oldz to newz
        """
        newz = str(newz)
        oldz = str(oldz)
        if newz != oldz:
            self.topic_d[d][oldz] += -1
            self.topic_w[oldz][intern(w)] += -1
            self.topic_wsum[oldz] += -1

            self.delta_topic_w[oldz][intern(w)] += -1
            self.delta_topic_wsum[oldz] += -1

            self.topic_d[d][newz] += 1
            self.topic_w[newz][intern(w)] += 1
            self.topic_wsum[newz] += 1

            self.delta_topic_w[newz][intern(w)] += 1
            self.delta_topic_wsum[newz] += 1

            d.assignment[i] = newz

        self.check_resync()



class DistributedLDA:
    def __init__(self, redis_instance, topics=50, alpha=0.1, beta=0.1):
        self.model = RedisLDAModelCache(redis_instance, topics)
        self.topics = topics
        self.beta = beta
        self.alpha = alpha
        self.V = 10000

    def insert_new_document(self, d):
        # sys.stderr.write('Inserting [%s]\n' % d.name)
        self.model.documents.append(d)
        d.assignment = []
        for w in d.dense:
            self.model.add_d_w(d, w, z=random.randint(0, self.topics))

    # @timed
    def cache_params(self, d):
        """
        Cache the Gibbs update math for this document
        """
        dndm1 = log(self.alpha*self.topics + d.nd - 1) 
        dnd   = log(self.alpha*self.topics + d.nd) 
        tdm1 = defaultdict(float)
        td   = defaultdict(float)
        m = self.model
        for tz in range(self.topics):
            tz = str(tz)
            if m.topic_wsum[tz] > 0 and m.topic_d[d][tz] > 0:
                tdm1[tz] = -log(self.beta*self.V + m.topic_wsum[tz] - 1) + log(self.alpha + m.topic_d[d][tz] - 1) - dndm1
            td[tz]   = -log(self.beta*self.V + m.topic_wsum[tz]) + log(self.alpha + (m.topic_d[d][tz])) - dnd

        return tdm1, td

    # @timed
    def resample_document(self, d):
        tdm1, td = self.cache_params(d)
        m = self.model
        for i, (w,oldz) in enumerate(izip(d.dense, d.assignment)):
            oldz = str(oldz)
            lp = []
            for tz in range(self.topics):
                tz = str(tz)
                if tz == oldz:
                    assert m.topic_w[tz][w] > 0
                    lp.append(log(self.beta + m.topic_w[tz][w] - 1) + tdm1[tz])
                else:
                    lp.append(log(self.beta + m.topic_w[tz][w]) + td[tz])

            newz = sample_lp_mult(lp)
            self.model.move_d_w(w, d, i, oldz, newz)

    def iterate(self, iterations):
        self.model.pull_global_state()
        for iter in range(iterations):
            self.do_iteration(iter)
        return self

    @timed
    def do_iteration(self, iter):
        for d in self.documents:
            self.resample_document(d)
        sys.stderr.write('done iter=%d\n' % iter)

    @timed
    def load_initial_docs(self, options):
        sys.stderr.write('Loading document shard %d / %d...\n' % (options.this_shard, options.shards))
        processed = 0
        for line_no, line in enumerate(open(options.document)):
            if line_no % options.shards == options.this_shard:
                d = Document(line=line)
                self.insert_new_document(d)
                self.resample_document(d)
                processed += 1
                if processed % 1e5 == 0:
                    sys.stderr.write('... loaded %d documents [%s]\n' % (processed, d.name))
        return self


if __name__ == '__main__':
    from argparse import ArgumentParser 
    from multiprocessing import Pool
 
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 

    # XXX: this doesnt actually work yet
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use") 

    parser.add_argument("--topics", type=int, default=100, help="Number of topics to use") 
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")

    parser.add_argument("--document", type=str, required=True, help="File to load as document") 

    parser.add_argument("--shards", type=int, default=1, help="Shard the document file into this many") 
    parser.add_argument("--this_shard", type=int, default=0, help="What shard number am I")

    options = parser.parse_args(sys.argv[1:]) 

    sys.stderr.write('Running on %d cores\n' % options.cores)
    
    # Get redis host and port
    try:
        host, port = options.redis.split(':')
    except:
        host = options.redis
        port = 6379

    sys.stderr.write('connecting to host %s:%d\n' % (host, int(port)))
    R = redis.StrictRedis(host=host, port=int(port), db=options.redis_db)

    sys.stderr.write("XXX: currently assuming unique docnames\n")

    if options.cores > 1:
        assert False
        p = Pool(options.cores)
        p.map(f, enumerate(open(options.document)))

    DistributedLDA(R, topics=options.topics).load_initial_docs(options).iterate(options.iterations)
