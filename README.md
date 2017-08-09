# markovest

The markovest of Markov chains, this will support the following:

* Color-coding based on the origin of the opus, so you can see at a glance which generated opuses the resultant sentences come from.
* Setting thresholds based on a tf-idf metric from used opuses, allowing you to filter sentences based on how many significant keywords are used.
* Caching the chain for faster subsequent runs once the original chain has been generated.
