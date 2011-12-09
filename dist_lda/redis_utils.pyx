import redis
import sys
from contextlib import contextmanager
# use the threading version
from multiprocessing.dummy import Pool

# TODO: make this a real configuration
MaxSimultaneousRedisConnections = 4


"""
The execute methods here are for when you want to call execute and not care about the 
return results (i.e., assume everything worked)

TODO: can probably make this stuff significantly more robust
"""

@contextmanager
def transact_single(r):
    yield r.pipeline()

@contextmanager
def execute_single(r, transaction=True):
    pipe = r.pipeline(transaction=transaction)
    yield pipe
    pipe.execute()


@contextmanager
def transact_block(rs, transaction=True):
    yield [r.pipeline(transaction=transaction) for r in rs]

@contextmanager
def execute_block(rs, transaction=True):
    pipes = [r.pipeline(transaction=transaction) for r in rs]
    yield pipes

    if len(pipes) > 1:
        # The price we pay for using undocumented python features :(
        # lifted from: http://bugs.python.org/issue10015
        import threading, weakref
        if not hasattr(threading.current_thread(), "_children"):
            threading.current_thread()._children = weakref.WeakKeyDictionary()

        # Call each redis pipeline.execute in parallel 
        Pool(MaxSimultaneousRedisConnections).map(lambda pipe: pipe.execute(), pipes)
    else:
        pipes[0].execute()

def connect_redis_list(redises, db):
    redises = redises.split(',')
    assert len(redises) == 1  # support for multiple redises is still wonky (can't guarantee transactions)
    # Get redis host and port
    connections = []
    for r in redises:
        try:
            host, port = r.split(':')
        except:
            host = r
            port = 6379
        sys.stderr.write('connecting to redis at %s:%d\n' % (host, int(port)))
        connections.append(redis.StrictRedis(host=host, port=int(port), db=db))
    return connections
