import redis
import sys
from contextlib import contextmanager

@contextmanager
def transact(rs):
    yield [r.pipeline() for r in rs]

@contextmanager
def execute(rs, transaction=True):
    pipes = [r.pipeline(transaction=transaction) for r in rs]
    yield pipes
    [pipe.execute() for pipe in pipes]



def connect_redis_list(redises, db):
    redises = redises.split(',')
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
