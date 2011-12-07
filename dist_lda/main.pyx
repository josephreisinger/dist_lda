import sys
import random
from collections import defaultdict
from libc.math cimport log
from itertools import izip
from sampler import sample_cum_lp_mult, addLog
from redis_model_cache import RedisLDAModelCache
from utils import timed, open_or_gz
from document import Document

class DistributedLDA:
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
        m = self.model

        cdef double dndm1 = log(self.alpha*self.topics + d.nd - 1)
        cdef double dnd   = log(self.alpha*self.topics + d.nd) 
        cdef double betaV = self.beta*m.v.size()
        cdef int tz
        tdm1 = defaultdict(float)  # keep this one sparse for now
        # try preallocating td
        td = [0 for tz in range(self.topics)]
        for tz in range(self.topics):
            assert m.topic_wsum[tz] >= 0
            if m.topic_wsum[tz] > 0 and m.topic_d[tz][d.id] > 0:
                tdm1[tz] = -log(betaV + m.topic_wsum[tz] - 1) + log(self.alpha + m.topic_d[tz][d.id] - 1) - dndm1
            
            td[tz] = -log(betaV + m.topic_wsum[tz]) + log(self.alpha + (m.topic_d[tz][d.id])) - dnd

        return tdm1, td

    # @timed
    def resample_document(self, d):
        cdef int topics = self.topics
        cdef double s = 0
        cdef double newz
        cdef int tz
        m = self.model

        with m.topic_lock:
            # try preallocating lp
            cum_lp = [0 for tz in range(topics)]
            tdm1, td = self.cache_params(d)
            for i, (w,oldz) in enumerate(izip(d.dense, d.assignment)):
                s = 0
                for tz in range(topics):
                    if tz == oldz:
                        assert m.topic_w[tz][w] > 0
                        s = addLog(s, log(self.beta + m.topic_w[tz][w] - 1) + tdm1[tz])
                    else:
                        assert m.topic_w[tz][w] >= 0
                        s = addLog(s, log(self.beta + m.topic_w[tz][w]) + td[tz])
                    cum_lp[tz] = s

                newz = sample_cum_lp_mult(cum_lp, topics)
                self.model.move_d_w(w, d, i, oldz, newz)

                self.attempts += 1
                if newz != oldz:
                    self.swaps += 1

    def iterate(self, iterations=None):
        if iterations == None:
            iterations = self.options.iterations
        for iter in range(iterations):
            self.do_iteration(iter)
            self.model.finalize_iteration(iter)
        return self

    @timed("do_iteration")
    def do_iteration(self, iter):
        self.swaps, self.attempts = 0, 0
        for d in self.model.documents:
            self.resample_document(d)
        # Print out the topics
        for z in range(self.topics):
            sys.stderr.write('I: %d [TOPIC %d] :: %s\n' % (iter, z, ' '.join(['[%s]:%d' % (w,c) for c,w in self.model.topic_to_string(self.model.topic_w[z])])))
        self.resamples += 1 
        sys.stderr.write('|| DONE iter=%d resamples=%d pulls=%d pushes=%d (%d swaps %.4f%%)\n' % (iter, self.resamples, self.model.pulls, self.model.pushes, self.swaps, 100 * self.swaps / float(self.attempts)))

    @timed("load_initial_docs")
    def load_initial_docs(self):
        sys.stderr.write('Loading document shard %d / %d...\n' % (self.options.this_shard, self.options.shards))
        processed = 0
        for line_no, line in enumerate(open_or_gz(self.options.document)):
            # Process every line because we need to build the vocabulary
            d = Document(line=line, vocab=self.model.v)
            if line_no % self.options.shards == self.options.this_shard:
                self.insert_new_document(d)
                self.resample_document(d)
                processed += 1
                if processed % 1000 == 0:
                    sys.stderr.write('... loaded %d documents [%s]\n' % (processed, d.name))
        sys.stderr.write("Loaded %d docs from [%s]\n" % (processed, self.options.document))
        assert processed > 0 # No Documents!
        self.model.finished_loading_docs = True
        return self

