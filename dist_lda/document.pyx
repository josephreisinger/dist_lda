from collections import defaultdict
import itertools

class Vocabulary:
    def __init__(self):
        self.to_id = defaultdict(itertools.count(0).next)
        self.to_word = {}
    def get(self, w):
        self.to_word[self.to_id[w]] = w
        return self.to_id[w]
    def rev(self, id):
        assert id in self.to_word
        return self.to_word[id]
    def size(self):
        return len(self.to_id)

# Base class holding a document and assignment vecto
class Document:
    def __init__(self, name="NULL", line=None, vocab=None):
        self.name = name
        self.dense = []
        self.assignment = []
        self.words = set()
        self.nd = 0
        assert vocab is not None
        self.vocab = vocab
        if line is not None:
            self.parse_from_line(line)
        self.id = vocab.get(name)

    def parse_from_line(self, line):
        self.name, word_counts = self.parse_lda_line(line)
        self.dense = [self.vocab.get(w) for w,c in word_counts for cc in range(c)]
        self.words = set(self.dense)
        self.nd = len(self.dense)

    @staticmethod
    def parse_lda_line(line):
        head, _, tokens = line.partition('\t')
        tokens = [x.rpartition(':') for x in tokens.split('\t')]

        return head, [(w, int(c)) for w,_,c in tokens]


