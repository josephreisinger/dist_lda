import redis
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
topics = R.get('topics')
alpha = R.get('alpha')
beta = R.get('beta')

print '%s\t%s\t%s\t%s' % (model, topics, alpha, beta)

with transact(self.r) as pipe:
    for z in range(topics):
        pipe.hgetall(('w', z))
    for z, zz in enumerate(pipe.execute()):
        topic_w[z] = defaultdict(int)
        for w,v in zz.iteritems():
            v = int(v)
            assert v >= 0
            if v > 0:
                print '%s\t%s\t%d' % (z, w, v)



