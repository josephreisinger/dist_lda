import sys
from time import sleep
from dist_lda import dump_model, connect_redis_string
from argparse import ArgumentParser 


if __name__ == '__main__':
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis", type=str, default="localhost:6379", help="Host for redis server") 
    parser.add_argument("--write_every", type=int, default=10, help="How many iterations to wait between writes") 
    parser.add_argument("--poll_interval", type=int, default=600, help="How many seconds to wait between polls") 
    options = parser.parse_args(sys.argv[1:]) 

    R = connect_redis_string(options.redis, options.redis_db)

    current_iter = None
    while True:
        sleep(options.poll_interval)
        shards = R.get('shards')
        iter = R.get('iterations')

        # Guard against nothing having completed even a single iteration
        if not iter:
            continue

        if not current_iter or (iter - current_iter) >= shards:
            current_iter = iter
            dump_model(R)

