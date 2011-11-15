import redis
import sys
from argparse import ArgumentParser 

parser = ArgumentParser() 
parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
parser.add_argument("--redis_port", type=int, default=26379, help="Port for redis server")
parser.add_argument("--redis_host", type=str, default="streetpizza.cs.utexas.edu", help="Host for redis server") 
options = parser.parse_args(sys.argv[1:]) 

R = redis.StrictRedis(host=options.redis_host, port=options.redis_port, db=options.redis_db)
sys.stderr.write("flushing db\n")
R.flushdb()
