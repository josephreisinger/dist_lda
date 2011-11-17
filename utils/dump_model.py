import redis
import json
import sys
from lda import *
from collections import defaultdict
from argparse import ArgumentParser 

parser = ArgumentParser() 
parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 
options = parser.parse_args(sys.argv[1:]) 

# Get redis host and port
try:
    host, port = options.redis.split(':')
except:
    host = options.redis
    port = 6379
R = redis.StrictRedis(host=host, port=int(port), db=options.redis_db)

model= R.get('model')
topics = int(R.get('topics'))
alpha = float(R.get('alpha'))
beta = float(R.get('beta'))


d = {}
d['model'] = model
d['topics'] = topics
d['alpha'] = alpha
d['beta'] = beta
d['params'] = defaultdict(lambda: defaultdict(int))

with transact(R) as pipe:
    for z in range(topics):
        pipe.hgetall(('w', z))
    for z, zz in enumerate(pipe.execute()):
        for w,v in zz.iteritems():
            v = int(v)
            assert v >= 0
            if v > 0:
                d['params'][z][w] = v


print json.dumps(d)

