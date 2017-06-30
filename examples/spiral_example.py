import numpy as np
import math
import random

class SpiralSystem:
	# time
	dt = 0.001
	timeline = np.arange(0,0.5,dt)

	# environment parameters
	k = 1/0.01
	rand_perturb = 0.0001
	x_bias = 0

	# movement parameters
	v = 0.1
	w = 4*math.pi
	r = 0.005
	v_max = v + w*r

	# compliance parameters
	#k = 10/0.01

	# input and output format
	state_size = 7 	#x, y, z, Fx, Fy, Fz, t
	action_size = 3

	# initial state
	init_state = np.array([r, 0, 0, 0, 0, 0, 0])

	# normalization params
	state_mu = [0, 0, 0.025, 0, 0, 0, 0.25]
	state_sigma = [0.01, 0.01, 0.5, 1, 1, 1, 0.25]
	action_mu = [0, 0, 0]
	action_sigma = [0.005, 0.005, 0.005]
	reward_mu = [0]
	reward_sigma = [1]

	# noise
	noise_mu = np.asarray([x_bias,0,0])
	noise_sigma = np.asarray([rand_perturb]*3)

	# default policy
	pi_0_array = []
	for t in timeline:
		temp = np.asarray([r*math.cos(w*t), r*math.sin(w*t), v*t])
		pi_0_array.append(temp)

	# continuous default policy
	def pi_0(self, t):
		idx = (np.abs(self.timeline - t)).argmin()
		return self.pi_0_array[idx]

	# simulate one step
	def step(self, state, action=np.zeros(action_size)):
		time_state = np.asarray([state[6] + self.dt])

		pos_state_desired = self.pi_0(time_state[0]) + action #+ random.gauss(self.noise_mu, self.noise_sigma)
		pos_vec = pos_state_desired - state[0:3]
		pos_state = state[0:3] + pos_vec*np.clip(3*self.v_max*self.dt/np.linalg.norm(pos_vec), 0, 1)

		force_state = np.asarray([self.k*pos_state[0], self.k*pos_state[1], 0])
		state = np.concatenate([pos_state, force_state, time_state], axis=0)
		return state

