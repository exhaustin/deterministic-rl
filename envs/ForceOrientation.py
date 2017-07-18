# Simulation of an object orientation problem
#	System states X - position and orientation
#	Observations F - forces and torques
#	Actions dX - change of position and orientation
#
#	System: F = K(X - X_op)		(K and X_op are randomly initialized)
#	System dynamics: X[n+1] = X[n] + dX[n]

import numpy as np
import math
import random

class ForceOrientation:
	def __init__(self, seed=None):
		self.sys = ForceSystem(seed)

		# input and output format
		self.state_dim = 6		#x, y, z, Rx, Ry, Rz
		self.action_dim = 6		#dx, dy, dz, dRx, dRy, dRz
		self.observation_dim = 6 	#Fx, Fy, Fz, Mx, My, Mz

		# states
		self.init_state = np.array([self.sys.r, 0, 0, 0])
		self.state = self.init_state
		self.time = 0

		# normalization params
		self.observation_mu = np.zeros(self.observation_dim)
		self.observation_sigma = np.ones(self.observation_dim)
		self.action_mu = np.zeros(self.action_dim)
		self.action_sigma = np.ones(self.action_dim)
		self.reward_mu = [0]
		self.reward_sigma = [1]

	# Simulate system for one timestep
	def step(self, action=None):
		if action is None:
			action = np.zeros(self.action_dim)

		# Simulate
		new_state = self.sys.step(self.state, action)

		# Done?
		done = self.goal_func(new_state)

		# Reward
		reward = self.reward_func(self.state, action, new_state, done)

		# Update state and time
		self.state = new_state
		self.time += 1

		return self.observe(), reward, done
	
	# Done definition
	def goal_func(self, state=None):
		return self.time >= 500

	# Reward definition
	def reward_func(self, state, action, new_state, done):
		return np.linalg.norm(state)

	# Get observataions
	def observe(self, state=None):
		if state is None:
			state = self.state

		return self.sys.observe(state)

	# Reset environment
	def reset(self):
		self.state = np.random.rand(self.state_dim)
		self.time = 0
	
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
		return state

class ForceSystem:
	def __init__(self, seed=None):
		# random seed
		if seed is not None:
			np.random.seed(seed)

		# optimal position
		self.state_op = np.random.rand(6)

		# stiffness matrix
		self.k = 1.5	#diagonal terms bias
		self.K = np.random.rand(6,6)
		for i in range(6):
			self.K[i,i] += self.k
			
	# simulate system for one step
	def step(self, state, action):
		return state + action

	# obtain reaction force
	def observe(self, state):
		state_diff = state - self.state_op
		return np.matmul(self.K, state_diff)

