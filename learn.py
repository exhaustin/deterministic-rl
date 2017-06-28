from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('methods/')
from DDPG import DDPGLearner
sys.path.append('examples/')
from spiral_example import SpiralSystem

class AgentWrapper:
	def __init__(self, state_size, action_size):
		self.model = DDPGLearner(state_size, action_size,
			BATCH_SIZE = 50,
			TAU = 0.1,
			LRA = 0.0001,
			LRC = 0.001,
			GAMMA = 0.99,
			HIDDEN1 = 300,
			HIDDEN2 = 600,
			verbose = False
		)

	# Main language -> Agent language
	def shape(self, vec):
		return np.reshape(vec, [1,-1])

	# Agent language -> Main language
	def deshape(self, vec):
		return vec[0,:]

	def act(self, state):
		return self.deshape( self.model.act(self.shape(state)) )

	def learn(self, state, action, reward, new_state, done):
		state1 = self.shape(state)
		action1 = self.shape(action)
		new_state1 = self.shape(new_state)
		self.model.learn(state1, action1, reward, new_state1, done)

class EnvWrapper:
	state_size = 7
	action_size = 3
	init_state = np.zeros(state_size)

	# normalization params
	state_mu = [0, 0, 0.025, 0, 0, 0, 0] 
	state_sigma = [0.01, 0.01, 0.05, 1, 1, 1, 1]
	action_mu = [0, 0, 0]
	action_sigma = [0.005, 0.005, 0.005]
	reward_mu = [0]
	reward_sigma = [1]
	state_shape = [np.array(state_mu), np.array(state_sigma)]
	action_shape = [np.array(action_mu), np.array(action_sigma)]
	reward_shape = [np.array(reward_mu), np.array(reward_sigma)]

	def __init__(self):
		self.system = SpiralSystem()
		self.state = self.init_state

	def sysInfo(self):
		return self.state_size, self.action_size, self.system.timeline

	def reset(self):
		self.state = self.init_state

	# Env language -> Main language
	def normalize(self, vec, shape):
		return (vec - shape[0])/shape[1]

	# Agent language -> Main language
	def denormalize(self, vec, shape):
		return vec*shape[1] + shape[0]

	def step(self, action):
		# Normalize state and action
		action_real = self.denormalize(action, self.action_shape)
		state_real = self.denormalize(self.state, self.state_shape)

		# Apply input to system
		new_state_real = self.system.step(state_real, action_real)
		reward_real = self.reward_func(state, action_real, new_state_real)

		# Goal?
		done = self.goal_func(new_state_real)

		# Normalize outputs
		new_state = self.normalize(new_state_real, self.state_shape)
		reward = self.normalize(reward_real, self.reward_shape)

		# Update system state
		self.state = new_state

		return new_state, reward, done

	def reward_func(self, state, action, new_state):
		reward = -(new_state[3]**2 + new_state[4]**2)
		return reward

	def goal_func(self, state):
		return False

	def render(self, state):
		state_real = self.denormalize(state, self.state_shape)
		print('({0:.2f},{1:.2f})\tdist={2:.2f}'.format(100*state_real[0], 100*state_real[1], 100*(state_real[0]**2 + state_real[1]**2)**0.5))
		return 0


if __name__ == '__main__':
	# training parameters
	max_episodes = 10

	# create environment
	env = EnvWrapper()
	state_size, action_size, timeline = env.sysInfo()

	# create agent
	agent = AgentWrapper(state_size, action_size)

	# create log database
	state_log = np.empty([max_episodes, state_size, len(timeline)])

	# run system
	for i_ep in range(max_episodes):
		# "Haruki, reset."
		env.reset()

		# train
		for T in range(len(timeline)):
			# simulate system for 1 timestep
			state = env.state
			action = agent.act(state)
			new_state, reward, done = env.step(action)
			agent.learn(state, action, reward, new_state, done)

			# record data
			state_log[i_ep, :, T] = state

			# display
			print('ep={0:2d}, \tt={1:.3f}\t'.format(i_ep, timeline[T]), end='')
			env.render(state)

