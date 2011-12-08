import sys
import random
from collections import defaultdict
from libc.math cimport exp, log
from itertools import izip
from redis_model_cache import RedisLDAModelCache
from utils import timed, open_or_gz
from document import Document

  
cdef inline double addLog(double x, double y):
    if x == 0:
        return y
    elif y == 0:
        return x
    elif x-y > 16:
        return x
    elif x > y:
        return x + log(1 + exp(y-x))
    elif y-x > 16:
        return y
    else:
        return y + log(1 + exp(x-y))

# XXX: this version is O(log(n)) but requires cumulative weights as inputs
# adapted from:
# http://blog.scaron.info/index.php/2011/06/better-weighted-random-choice-with-sagecython/
cdef int sample_cum_lp_mult(list cum_lp, int L):
    cdef double l_cut = log(random.random()) + cum_lp[-1]
    cdef int start = 0, mid, stop = L
    # edge case
    if l_cut <= cum_lp[start]:
        return 0
    while start < stop - 1:
        mid = (start + stop) / 2
        if cum_lp[mid] <= l_cut:
            start = mid
        else:
            stop = mid
    return stop


class Sampler:
    def iterate(self, iterations=None):
        if iterations == None:
            iterations = self.options.iterations
        for iter in range(iterations):
            self.do_iteration(iter)
        return self

    def initialize(self):
        raise 'NYI'

    def do_iteration(self, iter):
        raise 'NYI'


class DistributedLDA(Sampler):
    def __init__(self, options):
        self.topics = options.topics
        self.model = RedisLDAModelCache(options)
        self.beta = options.beta
        self.alpha = options.alpha
        self.options = options

        # Record some stats on what's going on
        self.swaps = 0
        self.attempts = 0
        self.resamples = 0

    # @timed
    def resample_document(self, d):
        m = self.model
        cdef int topics = self.topics
        cdef double s = 0
        cdef int newz
        cdef int tz
        cdef int i
        cdef int did = d.id
        cdef double dndm1 = log(self.alpha*self.topics + d.nd - 1)
        cdef double dnd   = log(self.alpha*self.topics + d.nd) 
        cdef double betaV = self.beta*m.v.size()

        with m.topic_lock:
            cum_lp = [0 for i in range(topics)]
            for i, (w,oldz) in enumerate(izip(d.dense, d.assignment)):
                s = 0
                for tz in range(topics):
                    # Obviously, vast majority of time is spent in this block
                    if tz == oldz:
                        assert m.topic_w[tz][w] > 0
                        s = addLog(s, \
                                log(self.beta + m.topic_w[tz][w] - 1) \
                                - log(betaV + m.topic_wsum[tz] - 1) \
                                + log(self.alpha + m.topic_d[tz][did] - 1) \
                                - dndm1)
                    else:
                        assert m.topic_w[tz][w] >= 0
                        s = addLog(s, \
                                log(self.beta + m.topic_w[tz][w]) \
                                - log(betaV + m.topic_wsum[tz]) \
                                + log(self.alpha + m.topic_d[tz][did]) \
                                - dnd)
                    cum_lp[tz] = s

                newz = sample_cum_lp_mult(cum_lp, topics)
                self.model.move_d_w(w, d, i, oldz, newz)

                self.attempts += 1
                if newz != oldz:
                    self.swaps += 1

    @timed("do_iteration")
    def do_iteration(self, iter):
        self.swaps, self.attempts = 0, 0
        for d in self.model.documents:
            self.resample_document(d)
        self.model.finalize_iteration(iter)
        # Print out the topics
        for z in range(self.topics):
            sys.stderr.write('I: %d [TOPIC %d] :: %s\n' % (iter, z, ' '.join(['[%s]:%d' % (w,c) for c,w in self.model.topic_to_string(self.model.topic_w[z])])))
        self.resamples += 1 
        sys.stderr.write('|| DONE iter=%d resamples=%d pulls=%d pushes=%d (%d swaps %.4f%%)\n' % (iter, self.resamples, self.model.pulls, self.model.pushes, self.swaps, 100 * self.swaps / float(self.attempts)))

    @timed("initialize")
    def initialize(self):
        sys.stderr.write('Loading document shard %d / %d...\n' % (self.options.this_shard, self.options.shards))
        processed = 0

        disk_journal = self.model.try_load_disk_journal()

        for line_no, line in enumerate(open_or_gz(self.options.document)):
            # Process every line because we need to build the vocabulary
            d = Document(line=line, vocab=self.model.v)
            if line_no % self.options.shards == self.options.this_shard:
                d.build_rep()
                if disk_journal:
                    assert len(disk_journal[processed]) == len(d.dense)
                    self.model.insert_new_document(d, assignments=disk_journal[processed])
                else:
                    self.model.insert_new_document(d, assignments=[random.randint(0, self.topics) for x in d.dense])
                    self.resample_document(d)
                processed += 1
                if processed % 1000 == 0:
                    sys.stderr.write('... loaded %d documents [%s]\n' % (processed, d.name))
        sys.stderr.write("Loaded %d docs from [%s]\n" % (processed, self.options.document))
        assert processed > 0 # No Documents!
        self.model.post_initialize()
        return self

