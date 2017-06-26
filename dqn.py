import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Our hero
class DQN:
	def __init__(self, state_size, action_size, params=[0.95, 1, 0.1, 0.995, 0.001, 24]):
		# learning parameters
		self.gamma = params[0] #discount rate
		self.epsilon = params[1] #exploration rate
		self.epsilon_min = params[2]
		self.epsilon_decay = params[3]
		self.learning_rate = params[4]
		self.h_width = params[5]

		# input & output dimensions
		self.state_size = state_size
		self.action_size = action_size

		# initialize memory
		self.memory = deque(maxlen=2000)

		# initialize model
		self.model = self._build_model()
		#self.target_model = self._build_model()
		#self.update_target_model()

	def _build_model(self):
		# NN for DQL
		model = Sequential()
		model.add(Dense(self.h_width, input_dim=self.state_size, activation='relu'))
		model.add(Dense(self.h_width, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size) # returns random troll action
		act_values = self.model.predict(state)
		return np.argmax(act_values[0]) #returns best action

	def replay(self, batch_size):
		minibatch = random.sameple(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = reward
			if not done:
				target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.modelfit(state, target_f, epochs=1, verbose=0)
		
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def load(self, name):
		self.model.load_weights(name)

	def save(self, name):
		self.model.save_weights(name)



