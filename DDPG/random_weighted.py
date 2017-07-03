import numpy as np
import random

def find_interval(x, partition, endpoints=True):
	for i in range(0, len(partition)):
		if x < partition[i]:
			return i-1 if endpoints else i

	return -1 if endpoints else len(partition)

def weighted_choice(seq, weights):
	x = random.random()
	
	cum_weights = [0] + list(np.cumsum(weights))
	index = find_interval(x, cum_weights)

	return seq[index]

def weighted_sample(pop, weights, k):
	sample = []
	idx_sample = set()
	pop_idxlist = list(range(len(pop)))
	weights = list(weights)
	weights = [x / sum(weights) for x in weights]


	while len(idx_sample) < k:
		choice = weighted_choice(pop_idxlist, weights)
		index = pop_idxlist.index(choice)

		# choose between two methods
		if len(pop) < 20*k:
			idx_sample.add(choice)

			weights.pop(index)
			pop_idxlist.remove(choice)
			weights = [x / sum(weights) for x in weights]
		else:
			if choice not in idx_sample:
				idx_sample.add(choice)

	for idx in idx_sample:
		sample.append(pop[idx])

	return sample
