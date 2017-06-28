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
		self.n_experiences = 0
		self.buffer = deque()

	def getBatch(self, batch_size):
		# Randomly sample batch_size examples
		if self.num_experience < batch_size:
			return random.sample(self.buffer, self.n_experiences)
		else:
			return random.sample(self.buffer, batch_size)
	
	def size(self):
		return self.buffer_size

	def add(self, state, action, reward, new_state, done):
		experience = (state, action, reward, new_state, done)
		if self.n_experience < self.buffer_size:
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

class DDPGLearner:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LRA=0.0001,	#learning rate for actor
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
		mu = 0
		theta = 0.15
		sigma = 0.2
		OU = lambda x : theta*(mu - x) + sigma*np.random.randn(1)

		# Produce action
		action_original = actor.model.predict(state)
		action_noise = toggle_explore*epsilon*OU(action_original)

		return np.clip(action_original + action_noise, -1, 1)

	# Recieve reward and learn
	def learn(self, state, action, reward, new_state, done):
		# Save data in buffer
		buff.add(state, action, reward, new_state, done)

		# Extract batch
		batch = buff.getBatch(BATCH_SIZE)
		states = np.asarray([e[0] for e in batch])
		actions = np.asarray([e[1] for e in batch])
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.asarray([e[3] for e in batch])
		dones = np.asarray([e[4] for e in batch])

		# Criticize
		target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
		
		# Train networks
		y_t = np.copy(actions)
		for k in range(len(batch)):
			if dones[k]:
				y_t[k] = rewards[k]
			else:
				y_t[k] = rewards[k] + GAMMA*target_q_values[k]
	
		loss += critic.model.train_on_batch([states, actions], y_t)
		a_for_grad = actor.model.predict(states)
		grads = critic.gradients(states, a_for_grad)
		actor.train(states, grads)

		# Update target networks
		actor.target_train()
		critic.target_train()
