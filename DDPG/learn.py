import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DDPG import DDPGLearner

sys.path.append('../examples/')
from spiral_example import SpiralSystem

class EnvWrapper:
	def __init__(self):
		# create environment
		self.system = SpiralSystem()
		self.state_size = self.system.state_size
		self.action_size = self.system.action_size

		# normalization params
		state_shape = [self.system.state_mu, self.system.state_sigma]
		action_shape = [self.system.action_mu, self.system.action_sigma]
		reward_shape = [self.system.reward_mu, self.system.reward_sigma]

		# initialize system
		self.init_state = self.system.init_state
		self.state = self.init_state

	def sysInfo(self):
		return self.state_size, self.action_size, self.system.timeline

	def reset(self):
		self.state = self.init_state

	# Env language -> Main language
	def normalize(self, vec, shape):
		return (vec - shape[0])/shape[1]

	# Main language -> Env language
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
		state_real_cm = 100*state_real
		dist = 100*(state_real[0]**2 + state_real[1]**2)**0.5
		print('pos=({0:5.2f},{1:5.2f}), \tdist={2:4.2f}'.format(state_real_cm[0], state_real_cm[1], dist), end='') 
		return 0


if __name__ == '__main__':
	# training parameters
	max_episodes = 10

	# create environment
	env = EnvWrapper()
	state_size, action_size, timeline = env.sysInfo()

	# create agent
	agent = DDPGLearner(state_size, action_size,
		BATCH_SIZE = 25,
		TAU = 0.1,
		LRA = 0.0001,
		LRC = 0.001,
		GAMMA = 0.99,
		HIDDEN1 = 150,
		HIDDEN2 = 300
	)

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
			#action = agent.act(state)
			new_state, reward, done = env.step(action=np.array([0,0,0]))
			#loss = agent.learn(state, action, reward, new_state, done)

			# record data
			state_log[i_ep, :, T] = state

			# display
			#print('ep={0:2d}, \tt={1:.3f}, \t'.format(i_ep+1, timeline[T]), end='')
			#env.render(state)
			#print(', \tloss={}'.format(loss))

		# plot results
		fig = plt.figure()
		ax = fig.gca(projection='3d')

		x = state_log[0,0,:]
		y = state_log[0,1,:]
		z = state_log[0,2,:]
		ax.plot(x, y, z)

		plt.show()
