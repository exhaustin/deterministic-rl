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

class ActorNetwork:
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
		self.model, self.weights, self.state = self.create_actor_network(state_dim, action_dim)
		self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_dim, action_dim)

		self.action_gradient = tf.placeholder(tf.float32,[None, action_dim])
		self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
		grads = zip(self.params_grad, self.weights)

		self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
		self.sess.run(tf.global_variables_initializer())

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
		#print("Building actor model...")
		"""	
		model = Sequential([
			Dense(self.HIDDEN1_UNITS, input_dim=state_dim, activation='elu', kernel_initializer=TruncatedNormal(0.0,1e-4), name="S"),
			Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=TruncatedNormal(0.0,1e-4)),
			Dense(action_dim, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-4)),
		])
		"""
		S = Input(shape=[state_dim])
		h0 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-3))(S)
		h1 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-3))(h0)
		V = Dense(action_dim, activation='tanh', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-3))(h1)
		model = Model(inputs=S, outputs=V)

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
		self.sess.run(tf.global_variables_initializer())

	def gradients(self, states, actions):
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
		#print('Building critic model...')
		S = Input(shape=[state_dim])
		w1 = Dense(self.HIDDEN1_UNITS, activation='elu')(S)
		h1 = Dense(self.HIDDEN2_UNITS, activation='tanh')(w1)

		A = Input(shape=[action_dim])
		a1 = Dense(self.HIDDEN2_UNITS, activation='tanh')(A)

		h2 = layers.concatenate([h1,a1])
		h3 = Dense(self.HIDDEN2_UNITS, activation='elu')(h2)
		V = Dense(1, activation='linear')(h3) #output_dim = action_dim or 1?

		model = Model(inputs=[S,A], outputs=V)
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
	
		return model, A, S
