import heapq
from collections import defaultdict
from document import Vocabulary

class LDAModelCache(object):
    """
    Holds the state of an LDA model and implements basic operations for interacting
    """
    def __init__(self, options):
        self.topics = options.topics

        # Track the local model state
        self.topic_d = defaultdict(lambda: defaultdict(int))
        self.topic_w = defaultdict(lambda: defaultdict(int))
        self.topic_wsum = defaultdict(int)

        # Also track the deltas of the stuff we want to sync
        self.delta_topic_d = defaultdict(lambda: defaultdict(int))
        self.delta_topic_w = defaultdict(lambda: defaultdict(int))
        self.delta_topic_wsum = defaultdict(int)

        # Document state
        self.documents = []
        self.v = Vocabulary()

    def post_initialize(self):
        pass

    def insert_new_document(self, d, assignment_fn=None):
        self.documents.append(d)
        for i,w in enumerate(d.dense):
            self.add_d_w(w, d, i, z=assignment_fn())

    def add_d_w(self, w, d, i, z=None):
        """
        Add word w to document d
        """
        # XXX: only call this in the lock
        d.assignment[i] = z

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
        # XXX: should only call this inside the lock
        if newz != oldz:
            self.topic_d[oldz][d.id] += -1
            self.topic_w[oldz][w] += -1
            self.topic_wsum[oldz] += -1

            self.topic_d[newz][d.id] += 1
            self.topic_w[newz][w] += 1
            self.topic_wsum[newz] += 1

            self.delta_topic_d[oldz][d.id] += -1
            self.delta_topic_w[oldz][w] += -1
            self.delta_topic_wsum[oldz] += -1

            self.delta_topic_d[newz][d.id] += 1
            self.delta_topic_w[newz][w] += 1
            self.delta_topic_wsum[newz] += 1

            d.assignment[i] = newz

    def topic_to_string(self, topic, max_length=20):
        result = []
        for w,c in topic.iteritems():
            if len(result) > max_length:
                heapq.heappushpop(result, (c,self.v.rev(w)))
            else:
                heapq.heappush(result, (c,self.v.rev(w)))
        return heapq.nlargest(max_length, result)


