import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

from .networks.PolicyNet_v0 import PolicyNetwork
from .networks.ValueNet_v0 import ValueNetwork
from .misc.ReplayBuffer import ReplayBuffer

class PG_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LR=0.0001,	#learning rate
		GAMMA=0.99,	#discount factor
		HIDDEN1=150,
		HIDDEN2=300,
		EXPLORE=2000,
		BUFFER_SIZE=1000,
		verbose=True,
		):

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.BATCH_SIZE=BATCH_SIZE
		self.LR = LR
		self.TAU = TAU
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BUFFER_SIZE

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.1
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1
		self.last_reward = 0

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
		self.policy = PolicyNetwork(sess, state_dim, action_dim, LR, TAU, HIDDEN1, HIDDEN2)
		if verbose: print('Creating policy baseline network...')
		self.policy_baseline = PolicyNetwork(sess, state_dim, action_dim, LR, TAU, HIDDEN1, HIDDEN2)
		if verbose: print('Creating baseline network...')
		self.baseline = ValueNetwork(sess, state_dim, 5*LR, TAU, int(HIDDEN1/2), int(HIDDEN2/2))

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
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(len(x))

		# Produce action
		action_original = self.policy.predict(state)
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
		self.buff.add((state, action, reward, new_state, done, self.last_reward))
		self.last_reward = reward

		# Update statistics
		self.steps += 1
		self.rewards += reward

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0] for e in batch], axis=0)
		actions = np.concatenate([e[1] for e in batch], axis=0)
		rewards = np.asarray([e[2] for e in batch])
		new_states = np.concatenate([e[3] for e in batch], axis=0)
		dones = np.asarray([e[4] for e in batch])
		
		# Compute greedy action gradients
		action_grads = np.empty([batchsize, self.action_dim])
		old_actions = self.policy_baseline.predict(states)
		baselines = self.baseline.predict(states)
		for k in range(batchsize):
			action_grads[k,:] = (rewards[k] - baselines[k]) * (actions[k] - old_actions[k])

		# Update policy
		self.policy.train_on_grads(states, np.clip(action_grads, -0.01, 0.01))

		# Update baselines
		self.baseline.train_on_batch(states, rewards)
		self.policy_baseline.train_on_batch(states, actions)

		# Update target networks
		if self.TAU > 0:
			self.policy.target_train()
			self.baseline.target_train()
			self.policy_baseline.target_train()

		# Update statistics
		self.eps += 1
		self.ep_total_rewards.append(self.rewards)
		self.ep_total_steps.append(self.steps)

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
