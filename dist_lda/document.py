# Base class holding a document and assignment vector
class Document:
    def __init__(self, name="NULL", line=None):
        self.name = name
        self.dense = []
        self.assignment = []
        self.words = set()
        self.nd = 0
        if line is not None:
            self.parse_from_line(line)

    def parse_from_line(self, line):
        self.name, word_counts = self.parse_lda_line(line)
        self.dense = [intern(w) for w,c in word_counts for cc in range(c)]
        self.words = set(self.dense)
        self.nd = len(self.dense)


    @staticmethod
    def parse_lda_line(line):
        head, _, tokens = line.partition('\t')
        tokens = [x.rpartition(':') for x in tokens.split('\t')]

        return head, [(w, int(c)) for w,_,c in tokens]


