import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

#from .networks.ActorCritic_target_v0 import ActorNetwork, CriticNetwork
from .models.PolicyNet_v0 import PolicyNetwork
from .models.QValueNet_v0 import QValueNetwork
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
		EXPLORE=4000,
		BUFFER_SIZE=5000,
		verbose=True,
		prioritized=False
		):

		self.BATCH_SIZE=BATCH_SIZE
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BUFFER_SIZE
		self.prioritized = prioritized

		# Ornstein-Uhlenbeck Process
		self.mu_OU = 0
		self.theta_OU = 0.1
		self.sigma_OU = 0.2

		# Initialize variables
		self.epsilon = 1
		self.steps = 0

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

		if self.prioritized:
			self.buff = PrioritizedReplayBuffer(self.BUFFER_SIZE)
		else:	
			self.buff = ReplayBuffer(self.BUFFER_SIZE)

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
		action_original = self.actor.predict(state)
		action_noise = toggle_explore*self.epsilon*OU(action_original)
		
		# Record step
		self.steps += 1

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
		self.buff.add([(state, action, reward, new_state, done), None])

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0][0] for e in batch], axis=0)
		actions = np.concatenate([e[0][1] for e in batch], axis=0)
		rewards = np.asarray([e[0][2] for e in batch])
		new_states = np.concatenate([e[0][3] for e in batch], axis=0)
		dones = np.asarray([e[0][4] for e in batch])

		# Train critic
		target_q_values = self.critic.predict([new_states, self.actor.predict(new_states)])	
		y = np.empty([batchsize])
		loss = 0
		for k in range(len(batch)):
			if dones[k]:
				y[k] = rewards[k]
			else:
				y[k] = rewards[k] + self.GAMMA*target_q_values[k]	

		loss = self.critic.train_on_batch([states, actions], y)

		# Update loss in buffer for prioritized experience
		if self.prioritized:
			for e in batch:
				e[1] = (1-self.GAMMA)*e[1] + self.GAMMA*loss

		# Train actor
		a_for_grad = self.actor.model.predict(states)
		grads = self.critic.action_gradients(states, a_for_grad)
		self.actor.train_on_grads(states, np.clip(grads, -0.01, 0.01))

		# Update target networks
		self.actor.target_train()
		self.critic.target_train()

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
