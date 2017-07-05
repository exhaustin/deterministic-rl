import numpy as np
import math
import random

class SpiralPath:
	def __init__(self):
		self.sys = PathSystem()

		# input and output format
		self.state_dim = 4		#x, y, z, t
		self.action_dim = 3
		self.observation_dim = 7 	#x, y, z, Fx, Fy, Fz, t

		# states
		self.init_state = np.array([self.sys.r, 0, 0, 0])
		self.state = self.init_state
		self.time = 0.0

		# normalization params
		self.observation_mu = [0, 0, 0.025, 0, 0, 0, 0.25]
		self.observation_sigma = [0.01, 0.01, 0.5, 1, 1, 1, 0.25]
		self.action_mu = [0, 0, 0]
		self.action_sigma = [0.005, 0.005, 0.005]
		self.reward_mu = [0]
		self.reward_sigma = [1]

	# Simulate system for one timestep
	def step(self, action):
		# Simulate
		new_state = self.sys.step(self.state, action)
		self.state = new_state

		# Done?
		done = self.goal_func(new_state)

		# Reward
		reward = self.reward_func(self.state, action, new_state, done)

		# Update state and time
		self.state = new_state
		self.time = self.state[3]

		return self.observe(), reward, done
	
	# Done definition
	def goal_func(self, state=None):
		if state is None:
			state = self.state
		return state[3] >= 0.5

	# Reward definition
	def reward_func(self, state, action, new_state, done):
		new_ob_state = self.observe(new_state)
		reward = -(new_ob_state[3]**2 + new_ob_state[4]**2) - (1e-9)*np.linalg.norm(action)**2
		return reward

	# Get observataions
	def observe(self, state=None):
		if state is None:
			state = self.state

		pos_state = state[0:3]
		t_state = state[3:4]
		force_state = np.asarray([self.sys.k*pos_state[0], self.sys.k*pos_state[1], 0])

		observation = np.concatenate([pos_state, force_state, t_state], axis=0)

		return observation

	# Reset environment
	def reset(self):
		self.state = self.init_state
		self.time = 0.0
	
	# Get time of environment
	def getTime(self):
		return self.time

	# Get state of environment
	def getState(self):
		return self.state

	# Set state of environment
	def setState(self, state):
		self.state = state

	# Render environment
	def render(self, state=None):
		if state is None:
			state = self.state
		return 100*state[0:3]

class PathSystem:
	def __init__(self):
		# time
		self.dt = 0.001
		self.timeline = np.arange(0,0.5,self.dt)

		# environment parameters
		self.k = 1/0.01
		self.rand_perturb = 0.0001
		self.x_bias = 0

		# movement parameters
		self.vz = 0.1
		self.w = 4*math.pi
		self.r = 0.005
		self.v_max = self.vz + self.w*self.r

		# compliance parameters
		#self.k = 10/0.01

		# noise
		self.noise_mu = np.asarray([self.x_bias,0,0])
		self.noise_sigma = np.asarray([self.rand_perturb]*3)

		# default policy
		self.pi_0_array = []
		for t in self.timeline:
			temp = np.asarray([self.r*math.cos(self.w*t), self.r*math.sin(self.w*t), self.vz*t])
			self.pi_0_array.append(temp)


	# simulate system for one step
	def step(self, state, action=None):
		if action is None:
			action = np.zeros(self.action_size)

		t_state = np.asarray([state[3] + self.dt])
		t_idx = (np.abs(self.timeline - t_state)).argmin()

		pos_state_desired = self.pi_0_array[t_idx] + action #+ random.gauss(self.noise_mu, self.noise_sigma)
		pos_vec = pos_state_desired - state[0:3]
		pos_state = state[0:3] + pos_vec*np.clip(3*self.v_max*self.dt/np.linalg.norm(pos_vec), 0, 1)

		state = np.concatenate([pos_state, t_state], axis=0)
		return state

