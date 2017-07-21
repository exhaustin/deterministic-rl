import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

from .models.PolicyNet_v0 import PolicyNetwork
from .models.QValueNet_v0 import QValueNetwork
from .misc.ReplayBuffer import ReplayBuffer

class DDPG_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.5,	#target network hyperparameter
		LRA=0.0001,	#learning rate
		LRC=0.001,	#learning rate
		GAMMA=0.99,	#discount factor
		HIDDEN1=300,
		HIDDEN2=600,
		EXPLORE=2000,
		BUFFER_SIZE=20000,
		verbose=True,
		):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LRA = LRA
		self.LRC = LRC
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BUFFER_SIZE

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.1
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1

		# Statistics
		self.eps = 0
		self.steps = 0
		self.rewards = 0
		self.ep_total_rewards = []
		self.ep_total_steps = []

		# Tensorflow GPU optimization
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		K.set_session(sess)

		# Initialize actor and critic
		if verbose: print('Creating policy network...')
		self.actor = PolicyNetwork(sess, state_dim, action_dim, LRA, TAU, HIDDEN1, HIDDEN2)
		if verbose: print('Creating baseline network...')
		self.critic = QValueNetwork(sess, state_dim, action_dim, LRC, TAU, HIDDEN1, HIDDEN2)

		self.buff = ReplayBuffer(BUFFER_SIZE)

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

		# Ornstein-Uhlenbeck Process
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(1, self.action_dim)

		# Produce action
		action_original = self.actor.predict(state)
		action_noise = toggle_explore*self.epsilon*OU(action_original)
		
		# Clip, reshape and output
		action_out =  np.clip(action_original + action_noise, -1, 1)
		return self.denormalize(action_out[0,:], 'action')

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
		self.buff.add((state, action, reward, new_state, done))

		# Update statistics
		self.steps += 1
		self.rewards += reward

		# Perform policy update after an episode is complete
		loss = None

		if done:
			states = np.concatenate([e[0] for e in self.buff.buffer], axis=0)
			actions = np.concatenate([e[1] for e in self.buff.buffer], axis=0)
			rewards = np.asarray([e[2] for e in self.buff.buffer])
			new_states = np.concatenate([e[3] for e in self.buff.buffer], axis=0)
			dones = np.asarray([e[4] for e in self.buff.buffer])

			# Calculate on-policy Q-values
			q_values = np.empty(self.steps)
			target_q_values = self.critic.predict([states[1:], actions[1:]])
			for t in range(self.steps):
				if dones[t]:
					q_values[t] = rewards[t]
				else:
					q_values[t] = rewards[t] + self.GAMMA*target_q_values[t]

			""" #True on-policy discounted rewards
			q_values = np.empty(self.steps)
			r = 0
			for t in reversed(range(self.steps)):
				r = rewards[t] + self.GAMMA*r
				q_values[t] = r
			"""

			# Update critic
			loss = self.critic.model.fit([states, actions], q_values, batch_size=self.BATCH_SIZE, epochs=int(self.BATCH_SIZE/2), verbose=False)

			# Train actor
			a_for_grad = self.actor.model.predict(states)
			grads = self.critic.action_gradients(states, a_for_grad)
			self.actor.train_on_grads(states, np.clip(grads, -0.01, 0.01))

			# Update target networks
			if self.TAU > 0:
				self.actor.target_train()
				self.critic.target_train()

			# Update statistics
			self.eps += 1
			self.ep_total_rewards.append(self.rewards)
			self.ep_total_steps.append(self.steps)

			# Print training info
			if verbose:
				print('ep={}, \ttotal_steps={}, \tloss={}'.format(self.eps, ep_total_steps[self.eps], 0))

			# Reset for new episode
			self.steps = 0
			self.rewards = 0
			self.buff.erase()

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
