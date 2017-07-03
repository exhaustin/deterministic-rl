import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from DDPG import DDPGLearner

sys.path.append('../examples/')
from spiral_example import SpiralSystem

# Formats inputs to and outputs from agents, and maybe performs normalization
class AgentWrapper:
	def __init__(self, env):
		self.agent = DDPGLearner(env.ob_state_size, env.action_size,
			BATCH_SIZE = 25,
			TAU = 0.1,
			LRA = 0.0001,
			LRC = 0.001,
			GAMMA = 0.99,
			HIDDEN1 = 150,
			HIDDEN2 = 300
		)

		self.state_shape = env.ob_state_shape
		self.action_shape = env.action_shape
		self.reward_shape = env.reward_shape

	# Env language -> Agent language
	def normalize(self, vec, shape):
		return (vec - shape[0])/shape[1]

	# Agent language -> Env language
	def denormalize(self, vec, shape):
		return vec*shape[1] + shape[0]

	# Act once
	def act(self, state):
		# normalize state
		state_N = self.normalize(state, self.state_shape)

		# act
		action_N = self.agent.act(state_N)

		# denormalize and return action
		action = self.denormalize(action_N, self.action_shape)
		return action

	# Learn from one experience
	def learn(self, state, action, reward, new_state, done):
		# normalize
		state_N = self.normalize(state, self.state_shape)
		action_N = self.normalize(action, self.action_shape)
		reward_N = self.normalize(reward, self.reward_shape)
		new_state_N = self.normalize(new_state, self.state_shape)

		# learn
		loss = self.agent.learn(state_N, action_N, reward_N, new_state_N, done)

		return loss


# Keeps track of environment states and manages interactions between agent and environment
class EnvWrapper:
	def __init__(self):
		# create environment
		self.system = SpiralSystem()
		self.state_size = self.system.state_size
		self.ob_state_size = self.system.ob_state_size
		self.action_size = self.system.action_size
		self.timeline = self.system.timeline

		# normalization params
		self.ob_state_shape = [self.system.ob_state_mu, self.system.ob_state_sigma]
		self.action_shape = [self.system.action_mu, self.system.action_sigma]
		self.reward_shape = [self.system.reward_mu, self.system.reward_sigma]

		# initialize system
		self.init_state = self.system.init_state
		self.state = self.init_state

	# Returns observations for agents
	def observe(self, state=None):
		if state is None:
			state = self.state

		return self.system.observe(state)

	# Resets environment to initial state
	def reset(self):
		self.state = self.init_state

	# Simulates environment for 1 timestep
	def step(self, action):
		# Apply input to system
		new_state = self.system.step(self.state, action)
		reward = self.reward_func(self.state, action, new_state)

		# Goal?
		done = self.goal_func(new_state)

		# Update system state
		self.state = new_state

		return new_state, reward, done

	# Definition of reward
	def reward_func(self, state, action, new_state):
		new_ob_state = self.system.observe(new_state)
		reward = -(new_ob_state[3]**2 + new_ob_state[4]**2) - (1e-9)*np.linalg.norm(action)**2
		return reward

	# Definition of whether goal is reached
	def goal_func(self, state=None):
		if state is None:
			state = self.state

		return False

	# Renders the environment
	def render(self, state=None):
		if state is None:
			state = self.state
		
		state_cm = 100*state
		return state_cm


if __name__ == '__main__':
	# training parameters
	max_episodes = 10

	# create environment
	env = EnvWrapper()
	state_size = env.state_size
	ob_state_size = env.ob_state_size
	action_size = env.action_size
	timeline = env.timeline

	# create agent
	agent = AgentWrapper(env)

	# create log database
	state_log = np.empty([max_episodes, state_size, len(timeline)+1])

	# run system
	for i_ep in range(max_episodes):
		# "Haruki, reset."
		env.reset()
		state_log[i_ep, :, 0] = env.render()

		# train
		for T in range(len(timeline)):
			# simulate system for 1 timestep
			ob_state = env.observe()
			action = agent.act(ob_state)
			new_state, reward, done = env.step(action)
			new_ob_state = env.observe()

			# train agent
			loss = agent.learn(ob_state, action, reward, new_ob_state, done)

			# record data
			state_log[i_ep, :, T+1] = env.render()

			# display
			print('ep={0:2d}, \tt={1:.3f}, \t'.format(i_ep+1, timeline[T]), end='')
			state_real_cm = env.render()
			dist = (state_real_cm[0]**2 + state_real_cm[1]**2)**0.5
			print('pos=({0:5.2f},{1:5.2f}), \tdist={2:4.2f}'.format(state_real_cm[0], state_real_cm[1], dist), end='') 
			print(', \tloss={}'.format(loss))

	# plot results
	eps = [0,3,6,9]
	for i_ep in eps:
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_xlim([-0.6, 0.6])
		ax.set_ylim([-0.6, 0.6])
		ax.set_zlim([0, 5])

		x = state_log[i_ep,0,:]
		y = state_log[i_ep,1,:]
		z = state_log[i_ep,2,:]
		ax.plot(x, y, z)

		plt.show()
	
