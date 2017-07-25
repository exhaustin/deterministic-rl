import numpy as np
import math
import random

from .models.PolicyLinear import PolicyModel
from .models.QValueLinear_v2 import QValueModel
from .misc.ReplayBuffer import ReplayBuffer

class LDPG_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		LRA=0.0001,	#learning rate for actor
		LRC=0.001,	#learning rate for critic
		GAMMA=0.99,
		EXPLORE=4000,
		BUFFER_SIZE=1000,
		K_init=None,
		verbose=False,
		):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.BATCH_SIZE = BATCH_SIZE
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.02
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1
		self.steps = 0
		self.action_buff_size = 3
		self.action_buff = []

		# Initialize actor and critic
		if verbose: print('Creating actor model...')
		self.actor = PolicyModel(state_dim, action_dim, lr=LRA, K_init=K_init, regularizer=1e-3)

		if verbose: print('Creating critic network...')
		self.critic = QValueModel(state_dim, action_dim, lr=LRC, regularizer=1e-2)

		# Initialize buffer
		self.buff = ReplayBuffer(BUFFER_SIZE)

	# Choose action
	def act(self, state_in, toggle_filter=False, toggle_explore=True):
		# Record step
		self.steps += 1

		# env format -> agent format
		state = self.normalize(state_in, 'state')
		state = np.reshape(state, [-1,1])

		# Diminishing exploration
		if self.epsilon > 0 and self.EXPLORE > 0:
			self.epsilon = np.exp(-self.steps/self.EXPLORE)
		else:
			self.epsilon = 0

		# Ornstein-Uhlenbeck Process
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(self.action_dim,1)

		# Produce action
		action_original = self.actor.predict(state)
		action_noise = toggle_explore*self.epsilon*OU(action_original)
		
		# Clip
		action = np.clip(action_original + action_noise, -1, 1)
		#action = action_original + action_noise

		# Smooth using moving average
		if toggle_filter:
			self.action_buff.append(action)
			if len(self.action_buff) >= self.action_buff_size:
				action = np.mean(np.concatenate(self.action_buff, axis=1), axis=1)
				action = np.reshape(action, [1,-1])
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
		reward = float(reward)
		new_state = np.reshape(new_state, [-1,1])
		done = bool(done)

		# Save experience in buffer
		self.buff.add((state, action, reward, new_state, done))

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0] for e in batch], axis=1)
		actions = np.concatenate([e[1] for e in batch], axis=1)
		rewards = np.asarray([[e[2] for e in batch]])
		new_states = np.concatenate([e[3] for e in batch], axis=1)
		dones = np.asarray([[e[4] for e in batch]])

		# Train critic
		target_q_values = rewards + self.GAMMA * self.critic.predict([new_states, self.actor.predict(new_states)])
		loss = self.critic.train([states, actions], target_q_values)

		#a0 = self.actor.predict(state)

		# Train actor
		grads = self.critic.action_gradients(states, actions)
		self.actor.train_on_grads(states, grads)
		#self.actor.train_on_grads(states, np.clip(grads, -0.1, 0.1))

		# Test
		#a1 = self.actor.predict(state)
		#q0 = self.critic.predict([state, a0])
		#q1 = self.critic.predict([state, a1])
		#if q0 > q1:
		#	print('FUCK')
		#print(q0)
		#print(q1)

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
