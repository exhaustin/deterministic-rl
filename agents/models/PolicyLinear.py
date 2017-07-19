import numpy as np
import math

class PolicyModel:
	def __init__(self, state_dim, action_dim, 
		lr,		#learning rate
		K_init=None,
		b_init=None,
		):

		self.state_dim = state_dim
		self.action_dim = action_dim
		self.lr = lr

		# create the model
		if K_init is None:
			K_init = np.zeros([action_dim, state_dim])
		if b_init is None:
			b_init = np.zeros([action_dim, 1])

		self.K = K_init
		self.b = b_init

	def train_on_grads(self, states, action_grads):
		batchsize = states.shape[1]

		for i in range(self.action_dim):
			for j in range(self.state_dim):
				self.K[i,j] += self.lr * np.mean(np.multiply(states[j,:], action_grads[i,:]))
			#self.b[i,0] +=	self.lr * action_grads[i,0] 

	def predict(self, states):
		return np.matmul(self.K, states) + self.b
