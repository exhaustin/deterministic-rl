import numpy as np
import math

class QValueModel:
	def __init__(self, state_dim, action_dim,
			lr,
			epsilon=1e-3,
			regularizer=1e-3,
			toggle_adagrad=True,
	):
		self.lr = lr
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.toggle_adagrad = toggle_adagrad

		# create the model
		self.M = epsilon * np.random.rand(1, action_dim)
		self.N = epsilon * np.random.rand(1, state_dim)
		self.b = epsilon * np.random.rand(1,1)
		#self.b = np.zeros([1,1])

		# adagrad
		self.M_gradss = np.ones([1, state_dim])
		self.N_gradss = np.ones([1, action_dim])
		self.b_gradss = 1
		self.lr_denom = 1

		# regularizer
		self.reg = regularizer

	def action_gradients(self, states, actions):
		a_grads = self.N.T
		return a_grads

	def train_on_grads(self, inputs, q_grads):
		states = inputs[0]
		actions = inputs[1]
		#batchsize = states.shape[1]

		for i in range(self.state_dim):
			Mw_grads = states[i,:]
			m_grads = np.multiply(Mw_grads, q_grads[0,:])	
			if self.toggle_adagrad:
				self.M_gradss[0,i] = 0.99*self.M_gradss[0,i] + np.mean(m_grads**2)
				self.lr_denom = 0.05 + 0.95*self.M_gradss[0,i]**0.5
			self.M[0,i] += (self.lr/self.lr_denom) * (np.mean(m_grads) - self.reg*self.M[0,i])

		for j in range(self.action_dim):
			Nw_grads = actions[j,:]
			n_grads = np.multiply(Nw_grads, q_grads[0,:])
			if self.toggle_adagrad:
				self.N_gradss[0,j] = 0.99*self.N_gradss[0,j] + np.mean(n_grads**2)
				self.lr_denom = 0.05 + 0.95*self.N_gradss[0,j]**0.5
			self.N[0,j] += (self.lr/self.lr_denom) * (np.mean(n_grads) - self.reg*self.N[0,j])

		if self.toggle_adagrad:
			self.b_gradss = 0.99*self.b_gradss + np.mean(q_grads**2)
			self.lr_denom = 0.05 + 0.95*self.b_gradss**0.5
		self.b += (self.lr/self.lr_denom) * np.mean(q_grads)

	def train(self, inputs, q_targets):
		delta = q_targets - self.predict(inputs)
		self.train_on_grads(inputs, delta)

		return np.mean(np.absolute(delta))

	def predict(self, inputs):
		state = inputs[0]
		action = inputs[1]
		q = np.matmul(self.M, state) + np.matmul(self.N, action) + self.b
		return q
