import sys
from collections import defaultdict

def timed(name):
    def _timed(func): 
        import time 
        """ Decorator @timed logs some timing info """ 
        def wrapper(*arg, **kwargs): 
            t1 = time.time() 
            res = func(*arg, **kwargs) 
            t2 = time.time() 
            sys.stderr.write('TIMED %s took %0.3f ms\n' % (name, (t2-t1)*1000.0)) 
            return res
        return wrapper
    return _timed


def with_retries(retries, name):
    """
    decorator with_retries will keep trying to do func until it doesn't raise an error
    the decorator augments the return state with the additional success / failure
    """
    def _retry(func):
        def wrapper(*arg, **kwargs):
            retry = 0
            while retry < retries:
                try:
                    res = func(*arg, **kwargs) 
                    return (True, res)
                except Exception, e:
                    retry += 1
                    sys.stderr.write('[%s] exception on %s (retry=%d/%d)\n' % (e, name, retry, retries))
            sys.stderr.write('ERROR: %s failed after %d retries\n' % (name, retries))
            return (False, None)
        return wrapper
    return _retry


def open_or_gz(f):
    if f.endswith('gz'):
        import gzip
        return gzip.open(f)
    else:
        return open(f)



def copy_sparse_defaultdict_1(dd):
    r = defaultdict(int)
    for z, c in dd.iteritems():
        if c != 0:
            r[z] = c
    return r

def copy_sparse_defaultdict_2(dd):
    r = defaultdict(lambda: defaultdict(int))
    for z,v in dd.iteritems():
        r[z] = copy_sparse_defaultdict_1(dd[z])
    return r


# Will modify first argument
def merge_defaultdict_1(result, x):
    for z, c in x.iteritems():
        if result[z] + c < 0:
            sys.stderr.write('RZ=%d c=%d\n' % (result[z], c))
        result[z] += c
        assert result[z] >= 0  # merge caused negative :(
    return result

def merge_defaultdict_2(result, x):
    for z,v in x.iteritems():
        result[z] = merge_defaultdict_1(result[z], x[z])
    return result
