import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from agents.DDPG_SARSA import DDPG_Agent
from envs.SpiralPath_fonly import SpiralPath

if __name__ == '__main__':
	# training parameters
	max_episodes = 20

	# create environment
	env = SpiralPath()
	state_dim = env.state_dim
	observation_dim = env.observation_dim
	action_dim = env.action_dim

	# create agent
	agent = DDPG_Agent(observation_dim, action_dim,
		BATCH_SIZE = 50,
		LRA = 0.0001,
		LRC = 0.001,
		GAMMA = 0.3,
		HIDDEN1 = 150,
		HIDDEN2 = 300,
	)

	agent.peek(env)

	# create log database
	state_log = np.empty([max_episodes, 3, 500+1])

	# run system
	for i_ep in range(max_episodes):
		# "Haruki, reset."
		env.reset()
		state_log[i_ep, :, 0] = env.render()
		done = False

		# train
		T = 0
		while not done:
			T += 1
			# simulate system for 1 timestep
			observation = env.observe()
			action = agent.act(observation)
			new_observation, reward, done = env.step(action)

			# train agent
			loss = agent.learn(observation, action, reward, new_observation, done)

			# record data
			state_cm = env.render()
			state_log[i_ep, :, T] = state_cm

			# display
			print('ep={0:2d}, \tt={1:.3f}, \t'.format(i_ep+1, env.getTime()), end='')
			dist = (state_cm[0]**2 + state_cm[1]**2)**0.5
			print('pos=({0:5.2f},{1:5.2f}), \tdist={2:4.2f}'.format(state_cm[0], state_cm[1], dist), end='')
			print(', \tloss={}'.format(loss))

	# plot results
	eps = [0,6,12,19]

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
	
