import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from agents.DDPG import DDPG_Agent
from envs.SpiralPath import SpiralPath

if __name__ == '__main__':
	# training parameters
	max_episodes = 1

	# create environment
	env = SpiralPath()
	state_dim = env.state_dim
	observation_dim = env.observation_dim
	action_dim = env.action_dim


	# run system
	state_log = np.empty([max_episodes, 3, 500+1])

	for i_ep in range(max_episodes):
		# "Haruki, reset."
		state_log[i_ep, :, 0] = env.render()
		done = False

		# train
		T = 0
		while not done:
			T += 1
			# simulate system for 1 timestep
			observation = env.observe()
			new_observation, reward, done = env.step()

			# record data
			state_cm = env.render()
			state_log[i_ep, :, T] = state_cm

	# plot default path
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

	# plot optimal path
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_xlim([-0.6, 0.6])
	ax.set_ylim([-0.6, 0.6])
	ax.set_zlim([0, 5])

	x = np.zeros(501)
	y = np.zeros(501)
	z = np.linspace(0,5,num=501)
	ax.plot(x, y, z)

	plt.show()
