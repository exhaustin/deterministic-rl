import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import Sequential, Model
from keras.engine.traning import collect_trainable_weights
from keras.layers import Dense, Flatten, input, merge, Lambda
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ReinforceLearner:
	def __init_(self, session, state_size, action_size,
			BATCH_SIZE=50,
			TAU=0.1,
			LEARNING_RATE=0.001):
		self.sess = session
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE

		K.set_session(sess)

		# create the model
		self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)
		self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size)
		self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())

	def train(self, states, action_grads):
	
	def target_train(self):

	def create_actor_network(self, state_size, action_dim):
