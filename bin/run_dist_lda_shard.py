import sys
from dist_lda import DistributedLDA

if __name__ == '__main__':
    from argparse import ArgumentParser 
    from multiprocessing import Pool
 
    parser = ArgumentParser() 
    parser.add_argument("--redis_db", type=int, default=0, help="Which redis DB") 
    parser.add_argument("--redis_hosts", type=str, default="localhost:6379", help="List of redises hosts for holding the model state") 

    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use") 

    parser.add_argument("--topics", type=int, default=100, help="Number of topics to use") 
    parser.add_argument("--alpha", type=float, default=0.1, help="Topic assignment smoother")
    parser.add_argument("--beta", type=float, default=0.0001, help="Vocab smoother")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")

    parser.add_argument("--document", type=str, required=True, help="File to load as document") 

    parser.add_argument("--shards", type=int, default=1, help="Shard the document file into this many") 
    parser.add_argument("--this_shard", type=int, default=0, help="What shard number am I")

    options = parser.parse_args(sys.argv[1:]) 

    sys.stderr.write('Running on %d cores\n' % options.cores)
    
    sys.stderr.write("XXX: currently assuming unique docnames\n")

    options.shards = options.cores * options.shards # split up even more


    def run_local_shard(core_id):
        # The basic idea here is the multiply the number of shards by the number of cores and
        # split them up even more
        options.this_shard = options.this_shard * options.cores + core_id
        options.core_id = core_id
        sys.stderr.write('initialize core %d on shard %d\n' % (core_id, options.this_shard))
        DistributedLDA(options).load_initial_docs().iterate()

    if options.cores > 1:
        p = Pool(options.cores)
        p.map(run_local_shard, range(options.cores))
    else:
        run_local_shard(0)


