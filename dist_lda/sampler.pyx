import random
from libc.math cimport exp, log, floor

cpdef double addLog(double x, double y):
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


# XXX: this version is O(n) but doesn't require the distribution to be normalized first (or cumulative weights)
def OLD_sample_lp_mult(list lp, int length):
    cdef s = 0
    cdef i

    cdef double cut = random.random()

    for i in range(length):
        s = addLog(s, lp[i])
    for i in range(length):
        cut -= exp(lp[i] - s)
        if cut < 0:
            return i
    
    assert False
    return 0



# XXX: this version is O(log(n)) but requires cumulative weights as inputs
# adapted from:
# http://blog.scaron.info/index.php/2011/06/better-weighted-random-choice-with-sagecython/
def sample_cum_lp_mult(list cum_lp, int L):
    cdef double l_cut = log(random.random()) + cum_lp[-1]
    cdef int start = 0, mid, stop = L
    # edge case
    if l_cut <= cum_lp[start]:
        return 0
    while start < stop - 1:
        mid = (start + stop) / 2
        if cum_lp[mid] <= l_cut:
            start = mid
        else:
            stop = mid
    return stop

