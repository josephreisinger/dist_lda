# dist_lda

Lightweight python implementation of a distributed, collapsed gibbs sampler for
LDA. Uses redis to coordinate multiple nodes.

dist_lda uses a dirty transaction model where words might be missing from local
counts and totals in order to improve performance at the expense of further
approximating the markov chain. I'm confident that there is a convergence
proof, but its too large to fit in the margin of this README.


## TODO
* Model dumping subscription service
* Shard topics over multiple redis servers (redis_cluster?)
* enumerate strings or otherwise low-bit hash to reduce mem footprint
* invert topic->word hashes to be word->topic . This way each word string is only stored once in redis, at the cost of significantly more pipelining
* massive amount of benchmarking

## PERF
* The current performance bottleneck seems to be the redis server, since a ton of information is being swapped around. Anecdotally I've found one master can coordinate up to ~20 model shards before performance starts to degrade. Current work is to distribute the model across multiple redii (say, hashed by topic).


## BUGS
* If individual processes die and restart, you'll get duplicate zombie words in the global state; fixing this would require significant re-architecting and would probably be too slow



## LICENSE

   Copyright 2011 Joseph Reisinger

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

