from dist_lda import connect_redis_string
import sys
from argparse import ArgumentParser 

parser = ArgumentParser() 
parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 
options = parser.parse_args(sys.argv[1:]) 

R = connect_redis_string(options.redis, options.redis_db)
sys.stderr.write("flushing db\n")
R.flushdb()
