import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

from .networks.PolicyNet import PolicyNetwork
from .misc.ReplayBuffer import ReplayBuffer

class REINFORCE_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LR=0.0001,	#learning rate
		GAMMA=0.99,
		HIDDEN1=300,
		HIDDEN2=600,
		EXPLORE=2000,
		BUFFER_SIZE=2000,
		verbose=True,
		):

		self.BATCH_SIZE=BATCH_SIZE
		self.GAMMA = GAMMA
		self.EXPLORE = EXPLORE
		self.BUFFER_SIZE = BUFFER_SIZE

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
		if verbose: print('Creating policy network...')
		self.policy = PolicyNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR, HIDDEN1, HIDDEN2)

		self.buff = ReplayBuffer(BUFFER_SIZE)

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
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(1)

		# Produce action
		action_original = self.policy.model.predict(state)
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
		self.buff.add(state, action, reward, new_state, done)

		# Extract batch
		batch, batchsize = self.buff.getBatch(self.BATCH_SIZE)
		states = np.concatenate([e[0][0] for e in batch], axis=0)
		actions = np.concatenate([e[0][1] for e in batch], axis=0)
		rewards = np.asarray([e[0][2] for e in batch])
		new_states = np.concatenate([e[0][3] for e in batch], axis=0)
		dones = np.asarray([e[0][4] for e in batch])

		# REINFORCE: REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility
		r = 0
		discounted_rewards = np.empty([batchsize])
		for k in range(len(batch)):
			r = self.rewards[k] + self.discount_factor*r
			discounted_rewards[k] = r

		# Update policy
		self.policy.train(states, grads)



		# Train critic
		target_q_values = self.critic.target_model.predict([new_states, self.policy.target_model.predict(new_states)])	
		y = np.empty([batchsize])
		loss = 0
		for k in range(len(batch)):
			if dones[k]:
				y[k] = rewards[k]
			else:
				y[k] = rewards[k] + self.GAMMA*target_q_values[k]	

		loss = self.critic.model.train_on_batch([states, actions], y)

		# Update loss in buffer for prioritized experience
		if self.prioritized:
			for e in batch:
				e[1] = (1-self.GAMMA)*e[1] + self.GAMMA*loss

		# Train actor
		a_for_grad = self.actor.model.predict(states)
		grads = self.critic.gradients(states, a_for_grad)
		self.actor.train(states, grads)




		# Update target networks
		self.policy.target_train()

		# Print training info
		if verbose:
			print('steps={}, loss={}'.format(self.steps, loss))

		# Return loss
		return loss
