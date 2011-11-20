import sys

def timed(func): 
    import time 
    """ Decorator @timed logs some timing info """ 
    def wrapper(*arg, **kwargs): 
        t1 = time.time() 
        res = func(*arg, **kwargs) 
        t2 = time.time() 
        sys.stderr.write('TIMED %s took %0.3f ms\n' % (func.func_name, (t2-t1)*1000.0)) 
        return res
    return wrapper


def open_or_gz(f):
    if f.endswith('gz'):
        import gzip
        return gzip.open(f)
    else:
        return open(f)
