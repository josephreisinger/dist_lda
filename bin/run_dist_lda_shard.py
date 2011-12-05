import sys
from dist_lda import DistributedLDA

if __name__ == '__main__':
    from argparse import ArgumentParser 
 
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis_hosts", type=str, default="localhost:6379", help="List of redises hosts for holding the model state") 

    parser.add_argument("--topics", type=int, default=100, help="Number of topics to use") 
    parser.add_argument("--alpha", type=float, default=0.1, help="Topic assignment smoother")
    parser.add_argument("--beta", type=float, default=0.0001, help="Vocab smoother")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")

    parser.add_argument("--document", type=str, required=True, help="File to load as document") 

    parser.add_argument("--shards", type=int, default=1, help="Shard the document file into this many") 
    parser.add_argument("--this_shard", type=int, default=0, help="What shard number am I")

    options = parser.parse_args(sys.argv[1:]) 

    sys.stderr.write("XXX: currently assuming unique docnames\n")

    sys.stderr.write('running on shard %d\n' % (options.this_shard))
    DistributedLDA(options).load_initial_docs().iterate()

