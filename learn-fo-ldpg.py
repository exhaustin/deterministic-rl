import numpy as np
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from agents.LDPG_QL import LDPG_Agent
from envs.ForceOrientation import ForceOrientation

if __name__ == '__main__':
	# training parameters
	max_episodes = 20

	# create environment
	env = ForceOrientation(seed=9487)
	state_dim = env.state_dim
	observation_dim = env.observation_dim
	action_dim = env.action_dim

	# create agent
	K_init = np.zeros([action_dim, state_dim])
	#"""
	for i in range(action_dim):
		if i < action_dim/2:
			K_init[i,i] = 0.2
		else:
			K_init[i,i] = 0.1
	#"""

	agent = LDPG_Agent(observation_dim, action_dim,
		BATCH_SIZE = 20,
		LRA = 1e-4,
		LRC = 1e-3,
		GAMMA = 0.3,
		EXPLORE = 8000,
		K_init = K_init,
	)

	agent.peek(env)

	# create log database
	state_log = np.empty([max_episodes, 1, 500+2])

	# run system
	for i_ep in range(max_episodes):
		# "Haruki, reset."
		env.reset(seed=9527)
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
			state_log[i_ep, :, T] = env.render()

			# display
			print('ep={}, \tt={}, \t'.format(i_ep+1, env.getTime()), end='')
			print('cost={0:3.5f}, \tloss={1:3.10f}'.format(-reward,loss))

	# plot results
	eps = [0,6,12,19]
	t = range(502)

	for ep in eps:
		plt.plot(t, state_log[ep,0,:])
		plt.axis([0, 500, 0, 2])
		plt.show()


