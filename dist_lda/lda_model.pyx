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
        self.delta_topic_wsum = defaultdict(int)

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
            self.delta_topic_wsum[z] += 1

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
            self.delta_topic_wsum[oldz] += -1

            self.delta_topic_d[d.id][newz] += 1
            self.delta_topic_w[w][newz] += 1
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


