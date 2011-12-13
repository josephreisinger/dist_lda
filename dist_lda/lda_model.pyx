import sys
import heapq
from collections import defaultdict
from document import Vocabulary

class LDAModelCache(object):
    """
    Holds the state of an LDA model and implements basic operations for interacting
    """
    def __init__(self, options):
        self.options = options
        self.topics = options.topics

        # Track the local model state
        self.topic_d = defaultdict(lambda: defaultdict(int))
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(int)

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))

        # Document state
        self.documents = []
        self.v = Vocabulary()

    def post_initialize(self):
        pass

    def insert_new_document(self, d, delta=True, assignments=None):
        self.documents.append(d)
        for i,w in enumerate(d.dense):
            self.add_d_w(w, d, i, delta=delta,  z=assignments[i])

    def add_d_w(self, w, d, i, delta=True, z=None):
        """
        Add word w to document d
        """
        # XXX: only call this in the lock
        d.assignment[i] = z

        self.topic_d[d.id][z] += 1
        self.topic_w[w][z] += 1
        self.topic_wsum[z] += 1
            
        # shouldn't increment the deltas if we've restarted from journaled state
        if delta:
            self.delta_topic_d[d.id][z] += 1
            self.delta_topic_w[w][z] += 1

    def move_d_w(self, w, d, i, oldz, newz):
        """
        Move w from oldz to newz
        """
        # XXX: should only call this inside the lock
        if newz != oldz:
            self.topic_d[d.id][oldz] += -1
            self.topic_w[w][oldz] += -1
            self.topic_wsum[oldz] += -1

            self.topic_d[d.id][newz] += 1
            self.topic_w[w][newz] += 1
            self.topic_wsum[newz] += 1

            self.delta_topic_d[d.id][oldz] += -1
            self.delta_topic_w[w][oldz] += -1

            self.delta_topic_d[d.id][newz] += 1
            self.delta_topic_w[w][newz] += 1

            d.assignment[i] = newz

    def topics_to_string(self, z, max_length=20):
        result = []
        for w,v in self.topic_w.iteritems():
            if len(result) > max_length:
                heapq.heappushpop(result, (v[z],self.v.rev(w)))
            else:
                heapq.heappush(result, (v[z],self.v.rev(w)))
        return heapq.nlargest(max_length, result)

    def dump_topics(self, iter):
        for z in range(self.topics):
            sys.stderr.write('I: %d [TOPIC %d] :: %s\n' % (iter, z, ' '.join(['[%s]:%d' % (w,c) for c,w in self.topic_to_string(z)])))

