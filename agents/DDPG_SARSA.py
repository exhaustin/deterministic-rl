import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

from .models.PolicyNet_v0 import PolicyNetwork
from .models.QValueNet_v0 import QValueNetwork
#from .models.QValueNet_dropout_v1 import QValueNetwork
from .misc.ReplayBuffer import ReplayBuffer
from .misc.PrioritizedReplayBuffer import PrioritizedReplayBuffer

class DDPG_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LRA=0.0001,	#learning rate for actor
		LRC=0.001,	#learning rate for critic
		GAMMA=0.99,
		HIDDEN1=150,
		HIDDEN2=300,
		EXPLORE=2000,
		BUFFER_SIZE=50,
		verbose=True
		):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.BATCH_SIZE=BATCH_SIZE
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BATCH_SIZE

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.02
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1
		self.steps = 0
		self.action_buff_size = 3
		self.action_buff = []

		# Tensorflow GPU optimization
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		sess = tf.Session(config=config)
		K.set_session(sess)

		# Initialize actor and critic
		if verbose: print('Creating actor network...')
		self.actor = PolicyNetwork(sess, state_dim, action_dim, LRA, TAU, HIDDEN1, HIDDEN2)

		if verbose: print('Creating critic network...')
		self.critic = QValueNetwork(sess, state_dim, action_dim, LRC, TAU, HIDDEN1, HIDDEN2)

		self.buff = ReplayBuffer(self.BUFFER_SIZE)

	# Choose action
	def act(self, state_in, toggle_filter=True, toggle_explore=True):
		# Record step
		self.steps += 1

		# env format -> agent format
		state = self.normalize(state_in, 'state')
		state = np.reshape(state, [1,-1])

		# Diminishing exploration
		if self.epsilon > 0:
			self.epsilon = np.exp(-self.steps/self.EXPLORE)
		else:
			self.epsilon = 0

		# Ornstein-Uhlenbeck Process
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(1, self.action_dim)

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
				action = np.reshape(action, [1,-1])
				self.action_buff.pop(0)

		# Reshape and output
		return self.denormalize(action[0,:], 'action')

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

		# Buffer content size
		batchsize = self.buff.count()

		# Update new_action in front of buffer
		if batchsize > 0:
			self.buff.buffer[-1][1] = action

		# Extract batch
		if batchsize > 1:
			states = np.concatenate([e[0][0] for e in self.buff.buffer], axis=0)
			actions = np.concatenate([e[0][1] for e in self.buff.buffer], axis=0)
			rewards = np.asarray([e[0][2] for e in self.buff.buffer])
			new_states = np.concatenate([e[0][3] for e in self.buff.buffer], axis=0)
			dones = np.asarray([e[0][4] for e in self.buff.buffer])
			new_actions = np.concatenate([e[1] for e in self.buff.buffer], axis=0)

		# Save experience in buffer
		self.buff.add([(state, action, reward, new_state, done), None])

		# End if buffer is empty
		if batchsize <= 1:
			return None

		# Train critic
		target_q_values = self.critic.predict([new_states, new_actions])	
		y = np.empty([batchsize])
		loss = 0
		for k in range(batchsize):
			if dones[k]:
				y[k] = rewards[k]
			else:
				y[k] = rewards[k] + self.GAMMA*target_q_values[k]	

		loss = self.critic.train_on_batch([states, actions], y)

		# Train actor
		a_for_grad = self.actor.model.predict(states)
		grads = self.critic.action_gradients(states, a_for_grad)
		self.actor.train_on_grads(states, np.clip(grads, -1e-3, 1e-3))

		# Update target networks
		self.actor.target_train()
		self.critic.target_train()

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
