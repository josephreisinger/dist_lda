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
        assert vocab is not None
        self.vocab = vocab

        self.name, self.word_counts = self.parse_lda_line(line)

        self.id = vocab.get(name)

    def build_rep(self):
        self.dense = [w for w,c in self.word_counts for cc in range(c)]
        self.assignment = [0 for w in self.dense]
        self.words = set(self.dense)
        self.nd = len(self.dense)

    def parse_lda_line(self, line):
        head, _, tokens = line.partition('\t')
        tokens = [x.rpartition(':') for x in tokens.split('\t')]

        return head, [(self.vocab.get(w), int(c)) for w,_,c in tokens]


