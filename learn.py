from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.append('methods/')
from DDPG import DDPGLearner
sys.path.append('examples/')
from spiral_example import SpiralSystem

class AgentWrapper:
	def __init__(self, state_size, action_size):
		return

	def act(self, state):
		return np.array([0,0,0])

	def remember(self, state, next_state, reward, done):
		return 0

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

	def normalize(self, vec, shape):
		return (vec - shape[0])/shape[1]
	
	def denormalize(self, vec, shape):
		return vec*shape[1] + shape[0]

	def step(self, action):
		action_real = self.denormalize(action, self.action_shape)
		state_real = self.denormalize(self.state, self.state_shape)

		next_state_real = self.system.step(state_real, action_real)
		reward_real = self.reward_func(state, next_state_real, action_real)

		next_state = self.normalize(next_state_real, self.state_shape)
		reward = self.normalize(reward_real, self.reward_shape)

		self.state = next_state

		return next_state, reward

	def reward_func(self, state, next_state, action):
		reward = -(next_state[3]**2 + next_state[4]**2)
		return reward

	def render(self, state):
		state_real = self.normalize(state, self.state_shape)
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
	for i_exec in range(max_episodes):
		# "Haruki, reset."
		env.reset()

		# train
		for T in range(len(timeline)):
			# simulate system for 1 timestep
			state = env.state
			action = agent.act(state)
			next_state, reward = env.step(action)
			agent.remember(state, action, reward, next_state)

			# record data
			state_log[i_exec, :, T] = state

