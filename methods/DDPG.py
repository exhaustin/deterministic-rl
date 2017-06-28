import numpy as np
import math
from collections import deque
import random
import tensorflow as tf
from keras import backend as K

from ActorCriticNet import ActorNetwork, CriticNetwork

class ReplayBuffer:
	def __init__(self, buffer_size):
		self.buffer_size = buffer_size
		self.num_experiences = 0
		self.buffer = deque()

	def getBatch(self, batch_size):
		# Randomly sample batch_size examples
		if self.num_experience < batch_size:
			return random.sample(self.buffer, self.num_experiences)
		else:
			return random.sample(self.buffer, batch_size)
	
	def size(self):
		return self.buffer_size

	def add(self, state, action, reward, new_state, done):
		experience = (state, action, reward, new_state, done)
		if self.num_experience < self.buffer_size:
			self.buffer.append(experience)
			self.num_experiences += 1
		else:
			self.buffer.popleft()
			self.buffer.append(experience)

	def count(self):
		return self.num_experiences

	def erase(self):
		self.buffer = deque()
		self.num_experiences = 0

class DDPGLearner:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LRA=0.0001	#learning rate for actor
		LRC=0.001,	#learning rate for critic
		GAMMA = 0.99
	):	
		# Parameters and variables
		np.random.seed(1337)
		self.EXPLORE = 100000
		self.reward = 0
		self.done = False
		self.step = 0
		self.epsilon = 1
		self.indicator = 0

		# Tensorflow GPU optimization
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		K.set_session(sess)

		# Initialize actor and critic
		actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
		critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
		buff = ReplayBuffer()

	# Choose action
	def act(self, state, toggle_explore=True):
		# Diminishing exploration
		if epsilon > 0:
			epsilon -= 1/self.EXPLORE
		else:
			epsilon = 0

		# Ornstein-Uhlenbeck Process
		OU = lambda x, mu, theta, sigma : theta*(mu - x) + sigma*np.random.randn(1)
		# Produce action
		action_original = actor.model.predict(state)
		action_noise = toggle_explore*epsilon*OU(action_original, 0, 0.15, 0.2)

		return np.clip(action_original + action_noise, -1, 1)

	# Recieve reward and learn
	def learn(self, state, next_state, action, reward):
		# Save data in buffer
		# Learn as a batch when buffer is full	
