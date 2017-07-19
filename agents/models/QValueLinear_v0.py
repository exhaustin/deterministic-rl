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
		self.b = epsilon * np.random.rand(1,1)

	def action_gradients(self, states, actions):
		return np.matmul(self.M.T, states)

	def train_on_grads(self, inputs, q_grads):
		state = inputs[0]
		action = inputs[1]

		for i in range(self.action_dim):
			for j in range(self.state_dim):
				self.M[i,j] += self.lr * q_grads[0,0] * state[j,0] * action[i,0]
		self.b -= self.lr * q_grads[0,0]

	def train(self, inputs, q_targets):
		delta = q_targets - self.predict(inputs)[0,0]
		self.train_on_grads(inputs, np.clip(delta, -0.01, 0.01))

		return abs(delta[0,0])

	def predict(self, inputs):
		state = inputs[0]
		action = inputs[1]
		return np.matmul(np.matmul(state.T, self.M) ,action) + self.b
