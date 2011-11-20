import random
from math import exp, log

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


