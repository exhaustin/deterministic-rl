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
		self.M = epsilon * np.random.rand(state_dim, action_dim)
		self.N = epsilon * np.random.rand(state_dim, state_dim)
		self.b = epsilon * np.random.rand(1,1)
		#self.b = np.zeros([1,1])

		# adagrad
		self.M_gradss = np.ones([state_dim, action_dim])
		self.N_gradss = np.ones([state_dim, state_dim])
		self.b_gradss = 1
		self.lr_denom = 1

		# regularizer
		self.reg = regularizer

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
				m_grads = np.multiply(Mw_grads, q_grads[0,:])	
				if self.toggle_adagrad:
					self.M_gradss[i,j] = 0.99*self.M_gradss[i,j] + np.mean(m_grads**2)
					self.lr_denom = 0.05 + 0.95*self.M_gradss[i,j]**0.5
				self.M[i,j] += (self.lr/self.lr_denom) * (np.mean(m_grads) - self.reg*self.M[i,j])

			for k in range(self.state_dim):
				Nw_grads = np.multiply(states[i,:], states[k,:])
				n_grads = np.multiply(Nw_grads, q_grads[0,:])
				if self.toggle_adagrad:
					self.N_gradss[i,k] = 0.99*self.N_gradss[i,k] + np.mean(n_grads**2)
					self.lr_denom = 0.05 + 0.95*self.N_gradss[i,k]**0.5
				self.N[i,k] += (self.lr/self.lr_denom) * (np.mean(n_grads) - self.reg*self.N[i,k])

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
		q = np.sum(np.multiply(np.matmul(state.T, self.M).T ,action), axis=0) + np.sum(np.multiply(np.matmul(state.T, self.N).T, state), axis=0) + self.b
		return q
