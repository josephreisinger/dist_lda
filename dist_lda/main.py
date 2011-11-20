import sys
import random
from collections import defaultdict
from math import log
from itertools import izip
from sampler import sample_lp_mult
from redis_model_cache import RedisLDAModelCache
from utils import timed, open_or_gz
from document import Document

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
            if iter % self.options.sync_every == 0:
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

