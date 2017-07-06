import numpy as np
import math
import random
import tensorflow as tf
from keras import backend as K

from .networks.Policy_v0 import PolicyNetwork
from .networks.Value_v0 import ValueNetwork
from .misc.ReplayBuffer import ReplayBuffer

class REINFORCE_Agent:
	def __init__(self, state_dim, action_dim,
		BATCH_SIZE=50,
		TAU=0.1,	#target network hyperparameter
		LR=0.0001,	#learning rate
		GAMMA=0.99,	#discount factor
		HIDDEN1=300,
		HIDDEN2=600,
		EXPLORE=2000,
		BUFFER_SIZE=20000,
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
		self.policy = PolicyNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LR, HIDDEN1, HIDDEN2)
		if verbose: print('Creating policy network...')
		self.baseline = ValueNetwork(sess, state_dim, BATCH_SIZE, TAU, LR*10, HIDDEN1, HIDDEN2)

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
		OU = lambda x : self.theta_OU*(self.mu_OU - x) + self.sigma_OU*np.random.randn(1)

		# Produce action
		action_original = self.policy.model.predict(state)
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
		self.buff.add((state, action, reward, new_state))

		# Update statistics
		self.steps += 1
		self.rewards += reward

		# Perform policy update after an episode is complete
		if done:
			states = np.concatenate([e[0][0] for e in buff.buffer], axis=0)
			actions = np.concatenate([e[0][1] for e in buff.buffer], axis=0)
			rewards = np.asarray([e[0][2] for e in buff.buffer])
			new_states = np.concatenate([e[0][3] for e in buff.buffer], axis=0)

			# REINFORCE: REward Increment = Nonnegative Factor x Offset Reinforcement x Characteristic Eligibility
			q = 0
			discounted_r = np.empty([1, self.steps])
			for t in reversed(range(len(batch))):
				q = self.rewards[t] + self.discount_factor*q
				discounted_r[t] = q
				baseline = self.baseline.model.predict(states[t])

				grads = LR*(discounted_r - baseline)

				# Update policy
				self.policy.train(states, grads)


			# Update baseline
			self.baseline.train(states, discounted_r)

			# Update statistics
			self.eps += 1
			self.ep_total_reward.append(rewards)
			self.ep_total_steps.append(steps)

			# Print training info
			if verbose:
				print('ep={}, \ttotal_steps={}, \tloss={}'.format(self.eps, ep_total_steps[self.eps], 0))

			# Reset for new episode
			steps = 0
			rewards = 0
			self.buff.erase()



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
