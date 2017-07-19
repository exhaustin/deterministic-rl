import numpy as np
import math
import random

from .models.PolicyLinear import PolicyModel
from .models.QValueLinear_v0 import QValueModel

class LDPG_Agent:
	def __init__(self, state_dim, action_dim,
		LRA=0.0001,	#learning rate for actor
		LRC=0.001,	#learning rate for critic
		GAMMA=0.99,
		EXPLORE=4000,
		K_init=None,
		verbose=False,
		):

		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.01
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1
		self.steps = 0
		self.action_buff_size = 3
		self.action_buff = []

		# Initialize actor and critic
		if verbose: print('Creating actor model...')
		self.actor = PolicyModel(state_dim, action_dim, lr=LRA, K_init=K_init)

		if verbose: print('Creating critic network...')
		self.critic = QValueModel(state_dim, action_dim, lr=LRC)

	# Choose action
	def act(self, state_in, toggle_filter=False, toggle_explore=True):
		# Record step
		self.steps += 1

		# env format -> agent format
		state = self.normalize(state_in, 'state')
		state = np.reshape(state, [-1,1])

		# Diminishing exploration
		if self.epsilon > 0:
			self.epsilon = np.exp(-self.steps/self.EXPLORE)
		else:
			self.epsilon = 0

		# Ornstein-Uhlenbeck Process
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(1)

		# Produce action
		action_original = self.actor.predict(state)
		action_noise = toggle_explore*self.epsilon*OU(action_original)
		
		# Clip
		action = np.clip(action_original + action_noise, -1, 1)

		# Smooth using moving average
		if toggle_filter:
			self.action_buff.append(action)
			if len(self.action_buff) >= self.action_buff_size:
				action = np.mean(np.concatenate(self.action_buff, axis=1), axis=1)
				self.action_buff.pop(0)

		# Reshape and output
		return self.denormalize(action[:,0], 'action')

	# Recieve reward and learn
	def learn(self, state_in, action_in, reward_in, new_state_in, done, verbose=False):
		# env format -> agent format
		state = self.normalize(state_in, 'state')
		action = self.normalize(action_in, 'action')
		reward = self.normalize(reward_in, 'reward')
		new_state = self.normalize(new_state_in, 'state')

		state = np.reshape(state, [-1,1])
		action = np.reshape(action, [-1,1])
		new_state = np.reshape(new_state, [-1,1])

		"""
		# Save experience in buffer
		self.buff.add([(state, action, reward, new_state, done), None])

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0][0] for e in batch], axis=0)
		actions = np.concatenate([e[0][1] for e in batch], axis=0)
		rewards = np.asarray([e[0][2] for e in batch])
		new_states = np.concatenate([e[0][3] for e in batch], axis=0)
		dones = np.asarray([e[0][4] for e in batch])
		"""

		# Train critic
		target_q_value = reward + self.GAMMA * self.critic.predict([new_state, self.actor.predict(new_state)])
		loss = self.critic.train([state, action], target_q_value)

		# Train actor
		grads = self.critic.action_gradients(state, action)
		self.actor.train_on_grads(state, np.clip(grads, -1e-3, 1e-3))

		# Print training info
		if verbose:
			print('steps={}, loss={}'.format(self.steps, loss))

		# Reset episodal stuff when done
		if done:
			action_buff = []

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
