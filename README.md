# dist_lda

Lightweight python implementation of a distributed, collapsed gibbs sampler for
LDA. Uses redis to coordinate multiple nodes. 

Documents and associated z-assignments are sharded across worker nodes (row sharding).
Model word/document counts are sharded across redis nodes by vocab (column sharding).

dist_lda uses a dirty transaction model where each shard's view of the global
state might lag behind the actual global state. This is essentially the model
of Newman et al "[Distributed Inference for Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)"

## Getting Started
First start your redis somewhere 

```
./src/redis-server redis.conf
```

Next start the model processing shards. These will divide up the input data into cores*shards pieces and divvy it out amongst all the cores.

```
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=0 --redis_hosts=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=1 --redis_hosts=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=2 --redis_hosts=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=3 --redis_hosts=host:6379 --sync_every=1
```

(I recommend pypy because currently its faster than python, and I'm not using numpy libraries)

Finally, optionally start up a listener to dump the model to disk every so often:

```
pypy listener.py --redis=server.path:6379 --write_every=1
```

This will generate a gzipped json representation of the model.

Note that you can pass multiple redis databases separating by comma, e.g.:

```
--redis_hosts=tygra:6379,panthro:6379,lion-o:6379
```

## Performance
* Performance bottleneck is communication. The ratio of worker shard updates to redis shards is critical, since there is a large amount of data transfer. Anecdotally I've found one master can coordinate up to ~20 model shards each with a few hundred MB of data before performance starts to degrade.
* Redis memory bottlenecks can be alleviated somewhat by sharding the model over multiple redis servers, lowering the number of worker shards, or lowering the rate of synchronization (say, once per 10 gibbs steps).


## Future Work
* Unblock / pipeline execute() statements; make them thread parallel.
* enumerate strings or otherwise low-bit hash to reduce mem footprint
* massive amount of benchmarking
* Support for sharded data files instead of single massive ones
* Automatic database flushing to avoid incorporating bits of stale models
* ROBUSTNESS / failure detection.
* Support for inference (duh)

## BUGS
* If individual processes die and restart, you'll get duplicate zombie words in the global state; fixing this would require some non trivial architectural work, and doesnt seem justified given the impact on the model.


## Contact
[Joseph Reisinger](http://www.cs.utexas.edu/~joeraii)
[@josephreisinger](http://www.twitter.com/josephreisinger)

## License

Apache 2.0. Please see LICENSE.md. All contents copyright (c) 2011, Joseph Reisinger.
