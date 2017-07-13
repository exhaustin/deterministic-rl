import numpy as np
import math
from keras import initializers
from keras.models import Sequential, Model
#from keras.engine.traning import collect_trainable_weights
from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class PolicyNetwork:
	def __init__(self, session, state_dim, action_dim, 
		lr,		#learning rate
		TAU=0,		#target network parameter
		HIDDEN1_UNITS=300,
		HIDDEN2_UNITS=600,
		BATCH_LOSS='mse',
		):

		self.sess = session
		self.lr = lr
		self.TAU = TAU
		self.HIDDEN1_UNITS = HIDDEN1_UNITS
		self.HIDDEN2_UNITS = HIDDEN2_UNITS
		self.BATCH_LOSS = BATCH_LOSS

		K.set_session(self.sess)

		# create the model
		self.model, self.state = self.create_actor_network(state_dim, action_dim)
		self.weights = self.model.trainable_weights

		if TAU > 0:
			self.target_model, self.target_state = self.create_actor_network(state_dim, action_dim)
		
		# gradient update path
		self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

		# initialize model
		self.sess.run(tf.global_variables_initializer())

	def create_actor_network(self, state_dim, action_dim):
		# tools
		K_INIT = initializers.TruncatedNormal(mean=0.0, stddev=1e-4)

		#print("Building actor model...")
		S = Input(shape=[state_dim])
		h0 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=K_INIT)(S)
		h1 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=K_INIT)(h0)
		V = Dense(action_dim, activation='tanh', kernel_initializer=K_INIT)(h1)
		model = Model(inputs=S, outputs=V)

		adam = Adam(lr=self.lr)
		model.compile(loss=self.BATCH_LOSS, optimizer=adam)

		return model, S

	def train_on_grads(self, states, action_grads):
		self.sess.run(self.optimize,
			feed_dict={
				self.state: states,
				self.action_gradient: action_grads
			}
		)

	def train_on_batch(self, states, actions_target):
		return self.model.train_on_batch(states, actions_target)

	def predict(self, states):
		if self.TAU > 0:
			return self.target_model.predict(states)
		else:
			return self.model.predict(states)

	def target_train(self):
		if self.TAU > 0:
			policy_weights = self.model.get_weights()
			policy_target_weights = self.target_model.get_weights()
			for i in range(len(policy_weights)):
				policy_target_weights[i] = self.TAU*policy_weights[i] + (1-self.TAU)*policy_target_weights[i]

			self.target_model.set_weights(policy_target_weights)
		else:
			print('Error: No target model.')
