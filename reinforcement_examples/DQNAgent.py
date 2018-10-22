import random
import numpy as np
from collections import deque
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, LeakyReLU, PReLU, Flatten, Conv2D, Dropout, Input, \
    Reshape
from tensorflow.python.keras import optimizers, backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers, regularizers

from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import TensorBoard


# play many episodes and remember some information from episodes randomly in memory
# and replay them
class DQNAgent:
    def __init__(self, state_size, action_size):
        np.random.seed(1)
        self.state_size = state_size
        self.action_size = action_size
        self.MEMORY_LEN = 590000
        self.memory = deque(maxlen=self.MEMORY_LEN)
        self.GAMMA = 0.95  # degree of feature effect (discount factor)
        self.EPSILON = 1.0  # exploration rate (in beginning)
        self.EPSILON_DECAY = 0.9999  # slowly shift exploration to exploitation
        self.EPSILON_MIN = 0.01
        self.LEARNING_RATE = 1e-2  # 0.001  # gradient descent rate 0.001
        self.EPOCHS = 1
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # input shape (samples, channels, rows, cols)
        model.add(Conv2D(32, (4, 4), strides=(2, 2), input_shape=(24, 10, 1),
                         kernel_initializer=initializers.glorot_uniform(), activation=activations.relu,
                         kernel_regularizer=regularizers.l2(0.01)))  # kernel initialize weights

        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activations.relu))

        model.add(Conv2D(128, (2, 2), strides=(1, 1), activation=activations.relu))

        # model.add(Dropout(0.5))
        model.add(Flatten())

        model.add(Dense(self.action_size, activation=activations.softmax))

        model.compile(loss=losses.categorical_crossentropy,  # loss='mse' losses.categorical_crossentropy
                      optimizer=optimizers.Adam(lr=self.LEARNING_RATE))  # RMSProb,Adam
        self.tensorBoard = TensorBoard('./logs/RLAgent', histogram_freq=0,
                                       write_graph=True, write_images=True)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # if len(self.memory) >= self.MEMORY_LEN:
        #     self.memory.popleft()
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print('act_value',(act_values[0]))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        miniBatch = random.sample(self.memory, batch_size)
        # states = []
        # targets = []
        for state, action, reward, next_state, done in miniBatch:
            target = reward  # if episode is done, target is out reward
            if not done:
                # estimate feature reward
                predict = self.model.predict(next_state)

                target = (reward + self.GAMMA * np.amax(predict))

            target_f = self.model.predict(state)  # predicted feature reward
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=self.EPOCHS, verbose=0)  # , callbacks=[self.tensorBoard])
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
