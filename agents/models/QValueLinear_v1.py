import numpy as np
import math

class QValueModel:
	def __init__(self, state_dim, action_dim,
			lr,
			epsilon=1e-3
	):
		self.lr = lr
		self.state_dim = state_dim
		self.action_dim = action_dim

		# create the model
		self.M = epsilon * np.random.rand(state_dim, action_dim)
		self.N = epsilon * np.random.rand(state_dim, state_dim)
		self.b = epsilon * np.random.rand(1,1)
		#self.b = np.zeros([1,1])

	def action_gradients(self, states, actions):
		a_grads = np.matmul(self.M.T, states)
		return a_grads

	def train_on_grads(self, inputs, q_grads):
		states = inputs[0]
		actions = inputs[1]
		#batchsize = states.shape[1]

		for i in range(self.state_dim):
			for j in range(self.action_dim):
				Mw_grads = np.multiply(states[i,:], actions[j,:])
				self.M[i,j] += self.lr * np.mean(np.multiply(Mw_grads, q_grads[0,:]))

			for k in range(self.state_dim):
				Nw_grads = np.multiply(states[i,:], states[k,:])
				self.N[i,k] += self.lr * np.mean(np.multiply(Nw_grads, q_grads[0,:]))

		self.b += self.lr * np.mean(q_grads)

	def train(self, inputs, q_targets):
		delta = q_targets - self.predict(inputs)
		self.train_on_grads(inputs, delta)

		return np.mean(np.absolute(delta))

	def predict(self, inputs):
		state = inputs[0]
		action = inputs[1]
		q = np.sum(np.multiply(np.matmul(state.T, self.M).T ,action), axis=0) + np.sum(np.multiply(np.matmul(state.T, self.N).T, state), axis=0) + self.b
		return q
