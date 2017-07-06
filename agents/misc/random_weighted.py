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

def weighted_sample(pop, weights, k, unique=True):
	sample = []
	pop_idxlist = list(range(len(pop)))
	w_0 = [True if x else False for x in weights]

	if unique and sum(w_0) < k:
		unique = False

	weights = [x / sum(weights) for x in weights]

	if unique:
		idx_sample = set()
	else:
		idx_sample = list()


	while len(idx_sample) < k:
		choice = np.random.choice(pop_idxlist, p=weights)
		if unique:
			idx_sample.add(choice)
		else:
			idx_sample.append(choice)

	for idx in idx_sample:
		sample.append(pop[idx])

	return sample
