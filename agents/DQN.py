import numpy as np
from itertools import product
import math
import random
import tensorflow as tf
from keras import backend as K

from .networks.QValue_target_v0 import QValueNetwork
from .misc.ReplayBuffer import ReplayBuffer

class DQN_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LR=0.0003,	#learning rate for critic
		GAMMA=0.99,
		HIDDEN1=300,
		HIDDEN2=600,
		EXPLORE=2000,
		BUFFER_SIZE=2000,
		ACTION_SIZE=10,
		verbose=True,
		prioritized=False
		):

		self.BATCH_SIZE=BATCH_SIZE
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BUFFER_SIZE
		self.prioritized = prioritized

		# Initialize variables
		self.epsilon = 1
		self.steps = 0

		# Tensorflow GPU optimization
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		K.set_session(sess)

		# Initialize actor and critic
		if verbose: print('Creating q-value network...')
		self.qnet = QValueNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR, HIDDEN1, HIDDEN2)

		self.buff = ReplayBuffer(BUFFER_SIZE)

		# Discretized action space
		sp = product(*[np.linspace(-1,1,num=ACTION_SIZE) for a in range(action_dim)])
		self.action_space = np.asarray([np.asarray(c) for c in sp])
		self.n_actions = self.action_space.shape[0]


	# Choose action
	def act(self, state_in, toggle_explore=True):
		# env format -> agent format
		state = self.normalize(state_in, 'state')
		state = np.reshape(state, [1,-1])

		# Diminishing exploration
		if self.epsilon > 0:
			self.epsilon -= 1/self.EXPLORE
		else:
			self.epsilon = 0
		
		if random.random() > self.epsilon:
			# Exploit
			q_values = self.qnet.target_model.predict([np.tile(state, [self.n_actions,1]), self.action_space])
			opt_idx = np.argmax(q_values)
			action = self.action_space[opt_idx,:]
		else:
			# Explore
			action = random.choice(self.action_space)

		# Record step
		self.steps += 1

		# Clip, reshape and output
		return self.denormalize(action, 'action')

	# Recieve reward and learn
	def learn(self, state_in, action_in, reward_in, new_state_in, done, verbose=False):
		# env format -> agent format
		state = self.normalize(state_in, 'state')
		action = self.normalize(action_in, 'action')
		reward = self.normalize(reward_in, 'reward')
		new_state = self.normalize(new_state_in, 'state')

		state = np.reshape(state, [1,-1])
		action = np.reshape(action, [1,-1])
		new_state = np.reshape(new_state, [1,-1])

		# Save experience in buffer
		self.buff.add([(state, action, reward, new_state, done), None])

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0][0] for e in batch], axis=0)
		actions = np.concatenate([e[0][1] for e in batch], axis=0)
		rewards = np.asarray([e[0][2] for e in batch])
		new_states = np.concatenate([e[0][3] for e in batch], axis=0)
		dones = np.asarray([e[0][4] for e in batch])

		# Train q-network
		random_action = random.choice(self.action_space)
		target_q_values = self.qnet.target_model.predict([new_states, np.tile(random_action, [self.BATCH_SIZE ,1])])
		y = np.empty([batchsize])
		loss = 0
		for k in range(len(batch)):
			if dones[k]:
				y[k] = rewards[k]
			else:
				y[k] = rewards[k] + self.GAMMA*target_q_values[k]	

		loss = self.qnet.model.train_on_batch([states, actions], y)

		# Update loss in buffer for prioritized experience
		if self.prioritized:
			for e in batch:
				e[1] = (1-self.GAMMA)*e[1] + self.GAMMA*loss

		# Update target networks
		self.qnet.target_train()

		# Print training info
		if verbose:
			print('steps={}, loss={}'.format(self.steps, loss))

		# Return loss
		return loss

	# Get information from the environment
	def peek(self, env):
		self.state_mu = env.observation_mu
		self.action_mu = env.action_mu
		self.reward_mu = env.reward_mu
		self.state_sigma = env.observation_sigma
		self.action_sigma = env.action_sigma
		self.reward_sigma = env.reward_sigma

	# Env language -> Agent language
	def normalize(self, vec, vtype):
		if vtype == 'state':
			mu = self.state_mu
			sigma = self.state_sigma
		elif vtype == 'action':
			mu = self.action_mu
			sigma = self.action_sigma
		elif vtype == 'reward':
			mu = self.reward_mu
			sigma = self.reward_sigma

		return (vec - mu)/sigma

	# Agent language -> Env language
	def denormalize(self, vec, vtype):
		if vtype == 'state':
			mu = self.state_mu
			sigma = self.state_sigma
		elif vtype == 'action':
			mu = self.action_mu
			sigma = self.action_sigma
		elif vtype == 'reward':
			mu = self.reward_mu
			sigma = self.reward_sigma

		return vec*sigma + mu
