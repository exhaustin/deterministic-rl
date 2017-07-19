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
	env = ForceOrientation()
	state_dim = env.state_dim
	observation_dim = env.observation_dim
	action_dim = env.action_dim

	# create agent
	K_init = np.zeros([action_dim, state_dim])
	for i in range(action_dim):
		if i < action_dim/2:
			K_init[i,i] = 1
		else:
			K_init[i,i] = 0.2

	agent = LDPG_Agent(observation_dim, action_dim,
		LRA = 1e-6,
		LRC = 1e-5,
		GAMMA = 0,
		EXPLORE = 4000,
		K_init = K_init,
	)

	agent.peek(env)

	# create log database
	state_log = np.empty([max_episodes, observation_dim, 500+2])

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
			F = env.render()
			state_log[i_ep, :, T] = F

			# display
			print('ep={}, \tt={}, \t'.format(i_ep+1, env.getTime()), end='')
			force = (F[0]**2 + F[1]**2 + F[2]**2)**0.5
			torque = (F[3]**2 + F[4]**2 + F[5]**2)**0.5
			print('force={0:5.2f}, \ttorque={1:4.2f}'.format(force, torque), end='')
			print(', \tloss={}'.format(loss))

	# plot results
	eps = [0,6,12,19]

