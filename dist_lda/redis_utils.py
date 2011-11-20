import redis
import sys
from contextlib import contextmanager

@contextmanager
def transact(r):
    pipe = r.pipeline()
    yield pipe

@contextmanager
def execute(r, transaction=True):
    pipe = r.pipeline(transaction=transaction)
    yield pipe
    pipe.execute()



def connect_redis_string(s, db):
    # Get redis host and port
    try:
        host, port = s.split(':')
    except:
        host = s
        port = 6379
    sys.stderr.write('connecting to redis at %s:%d\n' % (host, int(port)))
    return redis.StrictRedis(host=host, port=int(port), db=db)
