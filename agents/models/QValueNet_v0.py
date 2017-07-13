import numpy as np
import math
from keras import initializers
from keras import regularizers
from keras.models import Sequential, Model
#from keras.engine.traning import collect_trainable_weights
from keras import layers
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class QValueNetwork:
	def __init__(self, sess, state_dim, action_dim,
			lr,
			TAU=0,
			HIDDEN1_UNITS=300,
			HIDDEN2_UNITS=600,
			BATCH_LOSS='mse',
	):
		self.sess = sess
		self.TAU = TAU
		self.lr = lr
		self.action_dim = action_dim
		self.HIDDEN1_UNITS = HIDDEN1_UNITS
		self.HIDDEN2_UNITS = HIDDEN2_UNITS
		self.BATCH_LOSS = BATCH_LOSS

		K.set_session(sess)

		# create the model
		self.model, self.action, self.state = self.create_critic_network(state_dim, action_dim)
		self.weights = self.model.trainable_weights
		if TAU > 0:
			self.target_model, self.target_action, self.target_state = self.create_critic_network(state_dim, action_dim)

		# output action gradients
		self.action_grads = tf.gradients(self.model.output, self.action) # gradients for policy update

		# gradient update path
		self.q_gradient = tf.placeholder(tf.float32,[None, 1])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.q_gradient)
		grads = zip(self.params_grad, self.weights)
		self.optimize = tf.train.AdamOptimizer(lr).apply_gradients(grads)

		# initialize model
		self.sess.run(tf.global_variables_initializer())

	def create_critic_network(self, state_dim, action_dim):
		# tools
		K_INIT = initializers.TruncatedNormal(mean=0.0, stddev=1e-3)
		K_REG = regularizers.l2(1e-3)
		K_REG = None

		# for dropout
		#K.set_learning_phase(1)

		#print('Building critic model...')
		S = Input(shape=[state_dim])
		w1 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=K_INIT, kernel_regularizer=K_REG)(S)
		#w1_d = Dropout(rate=0.1)(w1)
		w2 = Dense(self.HIDDEN1_UNITS, activation='linear', kernel_initializer=K_INIT, kernel_regularizer=K_REG)(w1)
		#w2_d = Dropout(rate=0.1)(w2)

		A = Input(shape=[action_dim])
		a1 = Dense(self.HIDDEN2_UNITS, activation='linear', kernel_initializer=K_INIT, kernel_regularizer=K_REG)(A)
		#a1_d = Dropout(rate=0.1)(a1)

		h1 = layers.concatenate([w2,a1])
		h2 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=K_INIT, kernel_regularizer=K_REG)(h1)
		#h2_d = Dropout(rate=0.1)(h2)
		V = Dense(1, activation='linear')(h2) 
		model = Model(inputs=[S,A], outputs=V)

		adam = Adam(lr=self.lr)
		model.compile(loss=self.BATCH_LOSS, optimizer=adam)
	
		return model, A, S

	def action_gradients(self, states, actions):
		return self.sess.run(
			self.action_grads,
			feed_dict={
				self.state: states,
				self.action: actions
			}
		)[0]

	def train_on_grads(self, inputs, q_grads):
		self.sess.run(self.optimize,
			feed_dict={
				self.state: inputs[0],
				self.action: inputs[1],
				self.q_gradient: q_grads
			}
		)

	def train_on_batch(self, inputs, q_targets):
		return self.model.train_on_batch(inputs, q_targets)

	def predict(self, inputs):
		#K.set_learning_phase(0)
		if self.TAU > 0:
			return self.target_model.predict(inputs)
		else:
			return self.model.predict(inputs)
		#K.set_learning_phase(1)

	def target_train(self):
		if self.TAU > 0:
			q_weights = self.model.get_weights()
			q_target_weights = self.target_model.get_weights()
			for i in range(len(q_weights)):
				q_target_weights[i] = self.TAU*q_weights[i] + (1-self.TAU)*q_target_weights[i]
			self.target_model.set_weights(q_target_weights)
		else:
			print('Error: No target model.')
