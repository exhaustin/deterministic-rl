import numpy as np
from collections import deque
import random

from .random_weighted import weighted_sample

class PrioritizedReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.n_experiences = 0
		self.buffer = deque()

		self.max_loss = 0

	def getBatch(self, batch_size):
		# Update max loss
		self.max_loss = 0
		for e in self.buffer:
			if e[1] > self.max_loss:
				self.max_loss = e[1]

		# Sample
		if self.max_loss == 0:
			# Randomly sample batch_size examples
			if self.n_experiences < batch_size:
				return random.sample(self.buffer, self.n_experiences), self.n_experiences
			else:
				return random.sample(self.buffer, batch_size), batch_size
		else:
			# Sample with probability correlated to loss
			w = []
			for e in self.buffer:
				w.append(e[1])

			if self.n_experiences < batch_size:
				return weighted_sample(self.buffer, w, self.n_experiences, unique=False), self.n_experiences
			else:
				return weighted_sample(self.buffer, w, batch_size, unique=False), batch_size
	
	def size(self):
		return self.buffer_size

	def add(self, state, action, reward, new_state, done, loss=None):
		if loss is None:
			loss = self.max_loss

		experience = [(state, action, reward, new_state, done), loss]
		if self.n_experiences < self.buffer_size:
			self.buffer.append(experience)
			self.n_experiences += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def count(self):
		return self.n_experiences

	def erase(self):
		self.buffer = deque()
		self.n_experiences = 0
