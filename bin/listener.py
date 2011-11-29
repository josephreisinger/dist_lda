import sys
from time import sleep
from dist_lda import dump_model, connect_redis_list
from argparse import ArgumentParser 


if __name__ == '__main__':
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis_hosts", type=str, default="localhost:6379", help="List of redises hosts for holding the model state") 
    parser.add_argument("--write_every", type=int, default=10, help="How many iterations to wait between writes") 
    parser.add_argument("--poll_interval", type=int, default=600, help="How many seconds to wait between polls") 
    options = parser.parse_args(sys.argv[1:]) 

    rs = connect_redis_list(options.redis_hosts, options.redis_db)

    current_iter = None
    while True:
        sleep(options.poll_interval)
        shards = rs[0].get('shards')
        iter = rs[0].get('iterations')

        # Guard against nothing having completed even a single iteration
        if not iter:
            continue

        iter = int(iter)
        shards = int(shards)

        if not current_iter or (iter - current_iter) >= options.write_every*shards:
            current_iter = iter
            dump_model(rs)

