import sys
from dist_lda import dump_model, connect_redis_string
from argparse import ArgumentParser 


if __name__ == '__main__':
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 
    parser.add_argument("--write_every", type=int, default=10, help="How many iterations to wait between writes") 
    options = parser.parse_args(sys.argv[1:]) 

    R = connect_redis_string(options.redis, options.redis_db)
    dump_model(R)

