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
	def __init__(self, sess, state_dim, BATCH_SIZE, TAU, LEARNING_RATE,
			HIDDEN1_UNITS=300,
			HIDDEN2_UNITS=600
	):
		self.sess = sess
		self.BATCH_SIZE = BATCH_SIZE
		self.TAU = TAU
		self.LEARNING_RATE = LEARNING_RATE
		self.HIDDEN1_UNITS=HIDDEN1_UNITS
		self.HIDDEN2_UNITS=HIDDEN2_UNITS

		K.set_session(sess)

		# create the model
		self.model, self.state = self.create_value_network(state_dim)
		self.sess.run(tf.global_variables_initializer())

	def create_value_network(self, state_dim):
		#print('Building critic model...')
		S = Input(shape=[state_dim])
		h1 = Dense(self.HIDDEN1_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-4))(S)
		h2 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-4444))(h1)
		h3 = Dense(self.HIDDEN2_UNITS, activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=1e-4444))(h2)
		V = Dense(1, activation='linear')(h3) 

		model = Model(inputs=S, outputs=V)
		adam = Adam(lr=self.LEARNING_RATE)
		model.compile(loss='mse', optimizer=adam)
	
		return model, S
