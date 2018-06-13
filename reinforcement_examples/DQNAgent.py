# Cartpole DQN
import random
import numpy as np
from collections import deque
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import TensorBoard


# play many episodes and remember some information from episodes randomly in memory
# and replay them
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.MEMORY_LEN = 100000
        self.memory = deque(maxlen=self.MEMORY_LEN)
        self.GAMMA = 0.95  # degree of feature effect (discount factor)
        self.EPSILON = 1.0  # exploration rate (in beginning)
        self.EPSILON_DECAY = 0.995  # slowly shift exploration to exploitation
        self.EPSILON_MIN = 0.01
        self.LEARNING_RATE = 0.001  # gradient descent rate 0.001
        self.EPOCHS = 1
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='linear'))
        model.add(Dense(256, activation='linear'))
        model.add(Dense(128, activation='linear'))
        model.add(Dense(64, activation='linear'))
        model.add(Dense(self.action_size, activation='linear'))  # we dont want probability then use linear
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adam(lr=self.LEARNING_RATE))  # loss='mse' losses.categorical_crossentropy
        self.tensorBoard = TensorBoard('./logs/RLAgent', histogram_freq=0,
                                       write_graph=True, write_images=True)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print('act_values', act_values)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        miniBatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in miniBatch:
            target = reward  # if episode is done, target is out reward
            if not done:
                # estimate feature reward
                predict = self.model.predict(next_state)[0]
                # print(predict)
                target = (reward + self.GAMMA * np.amax(predict))
            # map that maximize feature reward to the current reward
            # map target  from current state to feature state
            target_f = self.model.predict(state)  # predicted feature reward
            target_f[0][action] = target
            # print('target', target)
            # fit model to train [input=state][output=predicted reward]
            # verbose is for show trainig process
            self.model.fit(state, target_f, epochs=self.EPOCHS, verbose=0)  # ,callbacks=[self.tensorBoard])
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
