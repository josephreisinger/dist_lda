# dist_lda

Lightweight python implementation of a distributed, collapsed gibbs sampler for
LDA. Uses redis to coordinate multiple nodes. Data is sharded by row.

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
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=0 --redis=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=1 --redis=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=2 --redis=host:6379 --sync_every=1
pypy bin/run_dist_lda_shard.py --topics=100 --document=data.gz --cores=2 --shards=4 --this_shard=3 --redis=host:6379 --sync_every=1
```

(I recommend pypy because currently its faster than python, and I'm not using numpy libraries)

Finally, optionally start up a listener to dump the model to disk every so often:

```
pypy listener.py --redis=server.path:6379 --write_every=1
```

This will generate a gzipped json representation of the model.

## Performance
* The current performance bottleneck seems to be the redis server, since a ton of information is being swapped around. Anecdotally I've found one master can coordinate up to ~20 model shards before performance starts to degrade. Current work is to distribute the model across multiple redii (say, hashed by topic).


## Future Work
* Shard topics over multiple redis servers (redis_cluster?)
* enumerate strings or otherwise low-bit hash to reduce mem footprint
* invert topic->word hashes to be word->topic . This way each word string is only stored once in redis, at the cost of significantly more pipelining
* massive amount of benchmarking
* Support for sharded data files instead of single massive ones
* Automatic database flushing to avoid incorporating bits of stale models
* Support for inference (duh)

## BUGS
* If individual processes die and restart, you'll get duplicate zombie words in the global state; fixing this would require some non trivial architectural work, and doesnt seem justified given the impact on the model.


## Contact
[Joseph Reisinger](http://www.cs.utexas.edu/~joeraii)
[@josephreisinger](http://www.twitter.com/josephreisinger)

## License

Apache 2.0. Please see LICENSE.md. All contents copyright (c) 2011, Joseph Reisinger.
