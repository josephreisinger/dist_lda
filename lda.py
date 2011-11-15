import random
from math import exp, log
import sys

# Base class holding a document but not its assignments
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
        for w,c in word_counts:
            for cc in range(c):
                self.dense.append(intern(w))
            self.words.add(intern(w))
        self.nd = len(self.dense)


    @staticmethod
    def parse_lda_line(line):
        head, _, tokens = line.partition('\t')
        tokens = [x.split(':') for x in tokens.split('\t')]

        return head, [(w, int(c)) for w,c in tokens]


def timed(func): 
    import time 
    """ 
    Decorator @timed logs some timing info 
    """ 
    def wrapper(*arg, **kwargs): 
        t1 = time.time() 
        res = func(*arg, **kwargs) 
        t2 = time.time() 
        sys.stderr.write('TIMED %s took %0.3f ms\n' % (func.func_name, (t2-t1)*1000.0)) 
        return res
    return wrapper

def addLog(x, y):
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


def sample_lp_mult(lp):
    cut = random.random()

    s = 0
    for ll in lp:
        s = addLog(s, ll)
    for i, ll in enumerate(lp):
        cut -= exp(ll - s)
        if cut < 0:
            return i
    
    assert False
    return 0
