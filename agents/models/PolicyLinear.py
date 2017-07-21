import numpy as np
import math

class PolicyModel:
	def __init__(self, state_dim, action_dim, 
		lr,		#learning rate
		K_init=None,
		b_init=None,
		toggle_adagrad=False,
		):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = lr
		self.toggle_adagrad = toggle_adagrad

		# create the model
		if K_init is None:
			K_init = np.zeros([action_dim, state_dim])
		self.K = K_init

		if b_init is None:
			b_init = np.zeros([action_dim, 1])
		#self.b = b_init

		self.K_gradss = np.ones([action_dim, state_dim])

	def train_on_grads(self, states, action_grads):
		#batchsize = states.shape[1]

		for i in range(self.action_dim):
			for j in range(self.state_dim):
				k_grads = np.multiply(states[j,:], action_grads[i,:])
				if self.toggle_adagrad:
					self.K_gradss[i,j] = 0.99*self.K_gradss[i,j] + 0.01*np.mean(k_grads**2)
					#self.K_gradss[i,j] = self.K_gradss[i,j] + np.mean(k_grads**2)
					self.K[i,j] += (self.lr / (0.1 + self.K_gradss[i,j]**0.5)) * np.mean(k_grads)
				else:
					self.K[i,j] += self.lr * np.mean(k_grads)
			#self.b[i,0] +=	self.lr * action_grads[i,0] 

	def predict(self, states):
		return np.matmul(self.K, states)# + self.b
