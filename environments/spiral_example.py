import numpy as np
import math
import random

class SpiralSystem:
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

		# input and output format
		self.state_size = 4	#x, y, z, t
		self.ob_state_size = 7 	#x, y, z, Fx, Fy, Fz, t
		self.action_size = 3

		# initial state
		self.init_state = np.array([self.r, 0, 0, 0])

		# normalization params
		self.ob_state_mu = [0, 0, 0.025, 0, 0, 0, 0.25]
		self.ob_state_sigma = [0.01, 0.01, 0.5, 1, 1, 1, 0.25]
		self.action_mu = [0, 0, 0]
		self.action_sigma = [0.005, 0.005, 0.005]
		self.reward_mu = [0]
		self.reward_sigma = [1]

		# noise
		self.noise_mu = np.asarray([self.x_bias,0,0])
		self.noise_sigma = np.asarray([self.rand_perturb]*3)

		# default policy
		self.pi_0_array = []
		for t in self.timeline:
			temp = np.asarray([self.r*math.cos(self.w*t), self.r*math.sin(self.w*t), self.vz*t])
			self.pi_0_array.append(temp)

	# get observataions
	def observe(self, state):
		pos_state = state[0:3]
		t_state = state[3:4]
		force_state = np.asarray([self.k*pos_state[0], self.k*pos_state[1], 0])

		ob_state = np.concatenate([pos_state, force_state, t_state], axis=0)
		return ob_state

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

