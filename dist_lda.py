import sys
import random
import heapq
from math import log
from collections import defaultdict
from itertools import izip

from redis_model_cache import RedisLDAModelCache
from lda_utils import *

class DistributedLDA:
    def __init__(self, options):
        self.model = RedisLDAModelCache(options)
        self.topics = options.topics
        self.beta = options.beta
        self.alpha = options.alpha
        self.V = options.vocab_size
        self.options = options

        # Record some stats on what's going on
        self.swaps = 0
        self.attempts = 0


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
            if m.topic_wsum[tz] > 0 and m.topic_d[tz][intern(d.name)] > 0:
                tdm1[tz] = -log(self.beta*self.V + m.topic_wsum[tz] - 1) + log(self.alpha + m.topic_d[tz][intern(d.name)] - 1) - dndm1
            td[tz]   = -log(self.beta*self.V + m.topic_wsum[tz]) + log(self.alpha + (m.topic_d[tz][intern(d.name)])) - dnd

        return tdm1, td

    # @timed
    def resample_document(self, d):
        tdm1, td = self.cache_params(d)
        m = self.model
        for i, (w,oldz) in enumerate(izip(d.dense, d.assignment)):
            lp = []
            for tz in range(self.topics):
                if tz == oldz:
                    assert type(tz) == type(oldz)
                    assert m.topic_w[tz][w] > 0
                    lp.append(log(self.beta + m.topic_w[tz][w] - 1) + tdm1[tz])
                else:
                    lp.append(log(self.beta + m.topic_w[tz][w]) + td[tz])

            newz = sample_lp_mult(lp)
            self.model.move_d_w(w, d, i, oldz, newz)

            self.attempts += 1
            if newz != oldz:
                self.swaps += 1

    def iterate(self, iterations=None):
        if iterations == None:
            iterations = self.options.iterations
        for iter in range(iterations):
            if iter % options.sync_every == 0:
                self.model.pull_global_state()
            self.do_iteration(iter)
            self.model.finalize_iteration(iter)
        return self

    @timed
    def do_iteration(self, iter):
        self.swaps, self.attempts = 0, 0
        for d in self.model.documents:
            self.resample_document(d)
        # Print out the topics
        for z in range(self.topics):
            sys.stderr.write('I: %d [TOPIC %d] :: %s\n' % (iter, z, ' '.join(['[%s]:%d' % (w,c) for c,w in self.model.topic_to_string(self.model.topic_w[z])])))
        sys.stderr.write('|| DONE core=%d iter=%d (%d swaps %.4f%%)\n' % (self.options.core_id, iter, self.swaps, 100 * self.swaps / float(self.attempts)))

    @timed
    def load_initial_docs(self):
        sys.stderr.write('Loading document shard %d / %d...\n' % (self.options.this_shard, self.options.shards))
        processed = 0
        for line_no, line in enumerate(open_or_gz(self.options.document)):
            if line_no % self.options.shards == self.options.this_shard:
                d = Document(line=line)
                self.insert_new_document(d)
                self.resample_document(d)
                processed += 1
                if processed % 1000 == 0:
                    sys.stderr.write('... loaded %d documents [%s]\n' % (processed, d.name))
        sys.stderr.write("Loaded %d docs from [%s]\n" % (processed, self.options.document))
        assert processed > 0 # No Documents!
        return self


if __name__ == '__main__':
    from argparse import ArgumentParser 
    from multiprocessing import Pool
 
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 

    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use") 

    parser.add_argument("--topics", type=int, default=100, help="Number of topics to use") 
    parser.add_argument("--alpha", type=float, default=0.1, help="Topic assignment smoother")
    parser.add_argument("--beta", type=float, default=0.1, help="Vocab smoother")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")

    parser.add_argument("--document", type=str, required=True, help="File to load as document") 
    parser.add_argument("--vocab_size", type=int, default=100000, help="Size of the document vocabulary") 

    parser.add_argument("--shards", type=int, default=1, help="Shard the document file into this many") 
    parser.add_argument("--this_shard", type=int, default=0, help="What shard number am I")

    # Resync intervals
    parser.add_argument("--sync_every", type=int, default=1, help="How many iterations should we wait to sync?")
    # Currently pull is every iteration 

    options = parser.parse_args(sys.argv[1:]) 

    sys.stderr.write('Running on %d cores\n' % options.cores)
    
    sys.stderr.write("XXX: currently assuming unique docnames\n")

    options.shards = options.cores * options.shards # split up even more

    def run_local_shard(core_id):
        # The basic idea here is the multiply the number of shards by the number of cores and
        # split them up even more
        options.this_shard = options.this_shard * options.cores + core_id
        options.core_id = core_id
        sys.stderr.write('initialize core %d on shard %d\n' % (core_id, options.this_shard))
        DistributedLDA(options).load_initial_docs().iterate()

    if options.cores > 1:
        p = Pool(options.cores)
        p.map(run_local_shard, range(options.cores))
    else:
        run_local_shard(0)


