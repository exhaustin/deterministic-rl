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
	w = 6
	r = 0.005

	# compliance parameters
	#k = 10/0.01

	# input and output format
	state_size = 7 	#x, y, z, Fx, Fy, Fz, t
	action_size = 3

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
		pos_state = self.pi_0(time_state) + action #+ random.gauss(self.noise_mu, self.noise_sigma)
		force_state = np.asarray([self.k*pos_state[0], self.k*pos_state[1], 0])
		state = np.concatenate([pos_state, force_state, time_state], axis=0)
		return state

