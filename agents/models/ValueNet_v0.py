import numpy as np
import math
from keras.initializers import TruncatedNormal
from keras.models import Sequential, Model
#from keras.engine.traning import collect_trainable_weights
from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ValueNetwork:
	def __init__(self, sess, state_dim, 
			lr,
			TAU=0,
			HIDDEN1_UNITS=300,
			HIDDEN2_UNITS=600,
			BATCH_LOSS='mse',
	):
		self.sess = sess
		self.TAU = TAU
		self.lr = lr
		self.HIDDEN1_UNITS = HIDDEN1_UNITS
		self.HIDDEN2_UNITS = HIDDEN2_UNITS
		self.BATCH_LOSS = BATCH_LOSS

		K.set_session(sess)

		# create the model
		self.model, self.state = self.create_critic_network(state_dim)
		self.weights = self.model.trainable_weights
		if TAU > 0:
			self.target_model, self.target_state = self.create_critic_network(state_dim)

		# gradient update path
		self.v_gradient = tf.placeholder(tf.float32,[None, 1])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.v_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

		# initialize model
		self.sess.run(tf.global_variables_initializer())

	def create_critic_network(self, state_dim):
		#print('Building critic model...')
		S = Input(shape=[state_dim])
		w1 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-2))(S)
		w2 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-2))(w1)
		h2 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-2))(w2)
		V = Dense(1, activation='linear')(h2) 
		model = Model(inputs=S, outputs=V)

		adam = Adam(lr=self.lr)
		model.compile(loss=self.BATCH_LOSS, optimizer=adam)
	
		return model, S

	def train_on_grads(self, states, v_grads):
		self.sess.run(self.optimize,
			feed_dict={
				self.state: states,
				self.v_gradient: v_grads
			}
		)

	def train_on_batch(self, states, v_targets):
		return self.model.train_on_batch(states, v_targets)

	def predict(self, states):
		if self.TAU > 0:
			return self.target_model.predict(states)
		else:
			return self.model.predict(states)

	def target_train(self):
		if self.TAU > 0:
			value_weights = self.model.get_weights()
			value_target_weights = self.target_model.get_weights()
			for i in range(len(value_weights)):
				value_target_weights[i] = self.TAU*value_weights[i] + (1-self.TAU)*value_target_weights[i]
			self.target_model.set_weights(value_target_weights)
		else:
			print('Error: No target model.')
