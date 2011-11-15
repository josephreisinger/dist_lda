import sys
import redis
import random
from math import log, exp
from collections import defaultdict
from itertools import izip
from contextlib import contextmanager

from lda import *


# XXX TODO: batch up changes instead of making them on the fly

@contextmanager
def transact(r):
    pipe = r.pipeline()
    yield pipe

@contextmanager
def execute(r, transaction=True):
    pipe = r.pipeline(transaction=transaction)
    yield pipe
    pipe.execute()

class RedisLDA:
    def __init__(self, redis_instance, topics=50, alpha=0.1, beta=0.1, sync_rate=10000):
        self.topics = topics
        self.r = redis_instance
        self.beta = beta
        self.alpha = alpha
        self.V = 10000
        self.sync_rate = sync_rate

        self.topic_d = defaultdict(lambda: defaultdict(float))
        self.topic_w = defaultdict(lambda: defaultdict(float))
        self.topic_wsum = defaultdict(float)

        self.documents = []
        self.resample_count = 0

    def add_d_w(self, pipe, d, w, z=None):
        z = str(z)
        pipe.hincrby(('w', z), w, 1)
        pipe.incr(('sum', z))
        pipe.hincrby(('d', d.name), z, 1)
        d.assignment.append(z)
        self.topic_d[d][z] += 1
        self.topic_w[z][intern(w)] += 1
        self.topic_wsum[z] += 1

    def move_d_w(self, pipe, w, d, i, z, newz):
        """
        Move w from z to newz
        """
        newz = str(newz)
        z = str(z)
        if newz != z:
            pipe.hincrby(('w', z), w, -1)
            pipe.hincrby(('d', d.name), z, -1)
            pipe.decr(('sum', z), 1)
            self.topic_d[d][z] += -1
            self.topic_w[z][intern(w)] += -1
            self.topic_wsum[z] += -1

            pipe.hincrby(('w', newz), w, 1)
            pipe.hincrby(('d', d.name), newz, 1)
            pipe.incr(('sum', newz), 1)
            self.topic_d[d][newz] += 1
            self.topic_w[newz][intern(w)] += 1
            self.topic_wsum[newz] += 1

            d.assignment[i] = z


    def insert_new_document(self, d):
        # sys.stderr.write('Inserting [%s]\n' % d.name)
        self.documents.append(d)
        with execute(self.r) as pipe:
            d.assignment = []
            for w in d.dense:
                self.add_d_w(pipe, d, w, z=random.randint(0, self.topics))

    @timed
    def resync_model(self):
        sys.stderr.write('skipping resync.\n')
        return
        sys.stderr.write('Resync model...\n')
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(float)

        with transact(self.r) as pipe:
            for z in range(self.topics):
                z = str(z)
                self.topic_w[z] = defaultdict(float)
                for w,v in pipe.hgetall(('w', z)).execute()[0].iteritems():
                    self.topic_w[z][w] = float(v)
                self.topic_wsum[z] = float(pipe.get(('sum', z)).execute()[0])

    # @timed
    def get_d(self, d):
        assert False
        topic_d = defaultdict(float)
        for z,v in self.r.hgetall(('d', d.name)).iteritems():
            topic_d[z] = float(v)
        return topic_d

    # @timed
    def cache_params(self, d):
        dndm1 = log(self.alpha*self.topics + d.nd - 1) 
        dnd   = log(self.alpha*self.topics + d.nd) 
        tdm1 = defaultdict(float)
        td   = defaultdict(float)
        for tz in range(self.topics):
            tz = str(tz)
            if self.topic_wsum[tz] > 0 and self.topic_d[d][tz] > 0:
                tdm1[tz] = -log(self.beta*self.V + self.topic_wsum[tz] - 1) + log(self.alpha + self.topic_d[d][tz] - 1) - dndm1
            td[tz]   = -log(self.beta*self.V + self.topic_wsum[tz]) + log(self.alpha + (self.topic_d[d][tz])) - dnd

        return dndm1, dnd, tdm1, td

    # @timed
    def update_model(self, d, tdm1, td):
        with execute(self.r, transaction=False) as pipe:
            for i, (w,z) in enumerate(izip(d.dense, d.assignment)):
                z = str(z)
                lp = []
                for tz in range(self.topics):
                    tz = str(tz)
                    if tz == z:
                        assert self.topic_w[tz][w] > 0
                        lp.append(log(self.beta + self.topic_w[tz][w] - 1) + tdm1[tz])
                    else:
                        lp.append(log(self.beta + self.topic_w[tz][w]) + td[tz])

                newz = sample_lp_mult(lp)
                self.move_d_w(pipe, w, d, i, z, newz)

    def resample_document(self, d):
        self.resample_count += 1
        # self.get_d(d)
        if self.resample_count > 0 and self.resample_count % self.sync_rate == 0:
            self.resync_model()

        # print d.name, topic_d
        # print topic_w

        dndm1, dnd, tdm1, td = self.cache_params(d)
        self.update_model(d, tdm1, td)

    def iterate(self, iterations):
        self.resync_model()
        for iter in range(iterations):
            self.do_iteration(iter)

    @timed
    def do_iteration(self, iter):
        for d in self.documents:
            self.resample_document(d)
        sys.stderr.write('done iter=%d\n' % iter)

    @timed
    def load_initial_docs(self, options):
        sys.stderr.write('Loading document shard %d...\n' % options.this_shard)
        for line_no, line in enumerate(open(options.document)):
            if line_no % options.shards == options.this_shard:
                d = Document(line=line)
                self.insert_new_document(d)
                self.resample_document(d)


if __name__ == '__main__':
    from argparse import ArgumentParser 
    from multiprocessing import Pool
 
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis_host", type=str, default="localhost:6379", help="Host for redis server") 
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
        host, port = options.redis_host.split(':')
    except:
        host = options.redis_host
        port = 6379

    sys.stderr.write('connecting to host %s:%d\n' % (host, int(port)))
    R = redis.StrictRedis(host=host, port=int(port), db=options.redis_db)
    sys.stderr.write("XXX: currently assuming unique docnames\n")
    RLDA = RedisLDA(R, topics=options.topics)


    def f((line_no, line)):
        if line_no % options.shards == options.this_shard:
            d = Document(line=line)
            RLDA.insert_new_document(d)
            RLDA.resample_document(d, line_no)
            # r = RedisIncrBuffer(max_size=100)
            # for w in d.dense:
            #     r.push((w, 1))
            # r.flush()

    if options.cores > 1:
        assert False
        p = Pool(options.cores)
        p.map(f, enumerate(open(options.document)))

    RLDA.load_initial_docs(options)
    RLDA.iterate(options.iterations)
