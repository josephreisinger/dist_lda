from dist_lda import connect_redis_list
import sys
from argparse import ArgumentParser 

parser = ArgumentParser() 
parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
parser.add_argument("--redis_hosts", type=str, default="localhost:6379", help="List of redises hosts for holding the model state") 
options = parser.parse_args(sys.argv[1:]) 

rs = connect_redis_list(options.redis_hosts, options.redis_db)

for r in rs:
    sys.stderr.write("flushing db\n")
    r.flushdb()
