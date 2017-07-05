import numpy as np
import math
from keras.initializers import TruncatedNormal
from keras.models import Sequential, Model
#from keras.engine.traning import collect_trainable_weights
from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.layers.merge import Add
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class PolicyNetwork:
	def __init__(self, session, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE,
		HIDDEN1_UNITS=300,
		HIDDEN2_UNITS=600
		):

		self.sess = session
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.HIDDEN1_UNITS=HIDDEN1_UNITS
		self.HIDDEN2_UNITS=HIDDEN2_UNITS

		K.set_session(self.sess)

		# create the model
		self.model, self.weights, self.state = self.create_policy_network(state_dim, action_dim)
		self.target_model, self.target_weights, self.target_state = self.create_policy_network(state_dim, action_dim)

		self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)

		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

	def train(self, states, update_grads):
		self.sess.run(self.optimize,
			feed_dict={
				self.state: states,
				self.action_gradient: update_grads
			}
		)

	def target_train(self):
		policy_weights = self.model.get_weights()
		policy_target_weights = self.target_model.get_weights()
		for i in range(len(policy_weights)):
			policy_target_weights[i] = self.TAU*policy_weights[i] + (1-self.TAU)*policy_target_weights[i]
		self.target_model.set_weights(policy_target_weights)

	def create_policy_network(self, state_dim, action_dim):
		#print("Building policy model...")
		S = Input(shape=[state_dim])
		h0 = Dense(self.HIDDEN1_UNITS, activation='elu', initializer=TruncatedNormal(0.0, 1e-4))(S)
		h1 = Dense(self.HIDDEN2_UNITS, activation='elu', initializer=TruncatedNormal(0.0, 1e-4))(h0)
		V = Dense(action_dim, activation='tanh')(h1)
		model = Model(inputs=S, outputs=V)

		return model, model.trainable_weights, S

