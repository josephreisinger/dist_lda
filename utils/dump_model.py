import redis
import json
import sys
import gzip
from lda_utils import *
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
d['document'] = document
d['w'] = defaultdict(lambda: defaultdict(int))
d['d'] = defaultdict(lambda: defaultdict(int))

doc_name = document.split('/')[-1]

with gzip.open('MODEL_%s-%s-T%d-alpha%.3f-beta%.3f.json.gz' % (model, doc_name, topics, alpha, beta), 'w') as f:
    with transact(R) as pipe:
        for z in range(topics):
            pipe.zrevrangebyscore(('w', z), float('inf'), 1, withscores=True)
        for z, zz in enumerate(pipe.execute()):
            for w,v in zz:
                d['w'][z][w] = int(v)

        # get documents
        for z in range(topics):
            pipe.zrevrangebyscore(('d', z), float('inf'), 1, withscores=True)
        for z, zz in enumerate(pipe.execute()):
            for doc,v in zz:
                d['d'][z][doc] = int(v)


    f.write(json.dumps(d))

