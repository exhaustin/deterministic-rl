import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import Sequential, Model
from keras.engine.traning import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K

class ActorNetwork:
	def __init_(self, session, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE,
			HIDDEN1_UNITS=300,
			HIDDEN2_UNITS=600
	):
		self.sess = session
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.HIDDEN1_UNITS=HIDDEN1_UNITS
		self.HIDDEN2_UNITS=HIDDEN2_UNITS

		K.set_session(sess)

		# create the model
		self.model, self.weights, self.state = self.create_actor_network(state_dim, action_dim)
		self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_dim, action_dim)
		self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.initialize_all_variables())

	def train(self, states, action_grads):
		self.sess.run(self.optimize,
			feed_dict={
				self.state: states,
				self.action_gradient: action_grads
			}
		)

	def target_train(self):
		actor_weights = self.model.get_weights()
		actor_target_weights = self.target_model.get_weights()
		for i in range(len(actor_weights)):
			actor_target_weights[i] = self.TAU*actor_weights[i] + (1-self.TAU)*actor_target_weights[i]
		self.target_model.set_weights(actor_target_weights)

	def create_actor_network(self, state_dim, action_dim):
		print("Building actor model...")
		S = Input(shape=[state_dim])
		h0 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
		h1 = Dense(self.HIDDEN2_UNITS, activateion='relu')(h0)
		V = Dense(action_dim, activation='tanh', init=lambda shape, name: normal(shape, scale=1e-4, name=name))(h1)
		model = Model(input=S, output=V)

		return model, model.trainable_weights, S

		 

class CriticNetwork:
	def __init__(self, sess, state_dim, action_dim, BATCH_SIZE, TAU, LEARNING_RATE,
			HIDDEN1_UNITS=300,
			HIDDEN2_UNITS=600
	):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.action_dim = action_dim
		self.HIDDEN1_UNITS=HIDDEN1_UNITS
		self.HIDDEN2_UNITS=HIDDEN2_UNITS

		K.set_session(sess)

		# create the model
		self.model, self.action, self.state = self.create_critic_network(state_dim, action_dim)
		self.target_model, self.target_action, self.target_state = self.create_critic_network(state_dim, action_dim)
		self.action_grads = tf.gradients(self.model.output, self.action) # gradients for policy update
		self.sess.run(tf.initialize_all_variables())

	def gradients(self, sstates, actions):
		return self.sess.run(
			self.action_grads,
			feed_dict={
				self.state: states,
				self.action: actions
			}
		)[0]

	def target_train(self):
		critic_weights = self.model.get_weights()
		critic_target_weights = self.target_model.get_weights()
		for i in range(len(critic_weights)):
			critic_target_weights[i] = self.TAU*critic_weights[i] + (1-self.TAU)*critic_target_weights[i]
		self.target_model.set_weights(critic_target_weights)

	def create_critic_network(self, state_dim, action_dim):
		print('Building critic model...')
		S = Input(shape=[state_dim])
		w1 = Dense(self.HIDDEN1_UNITS, activation='relu')(S)
		h1 = Dense(self.HIDDEN2_UNITS, activation='linear')(w1)

		A = Input(shape=[action_dim])
		a1 = Dense(self.HIDDEN2_UNITS, activation='linear')(A)

		h2 = merge([h1,a1], mode='sum')
		h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
		V = Dense(action_dim, activation='linear')(h3)

		model - Model(input=[S,A], output=V)
		adam = Adam(lr=self.LEARNING-RATE)
		model.compile(loss='mse', optimizer=adam)
	
		return model, A, S
