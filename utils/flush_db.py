import redis
import sys
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
R = redis.StrictRedis(host=options.host, port=options.port, db=options.redis_db)
sys.stderr.write("flushing db\n")
R.flushdb()
