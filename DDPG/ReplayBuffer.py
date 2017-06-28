import numpy as np
from collections import deque
import random

class ReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.n_experiences = 0
		self.buffer = deque()

	def getBatch(self, batch_size):
		# Randomly sample batch_size examples
		if self.n_experiences < batch_size:
			return random.sample(self.buffer, self.n_experiences), self.n_experiences
		else:
			return random.sample(self.buffer, batch_size), batch_size
	
	def size(self):
		return self.buffer_size

	def add(self, state, action, reward, new_state, done):
		experience = (state, action, reward, new_state, done)
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
