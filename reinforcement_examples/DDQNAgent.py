import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, LeakyReLU, PReLU, Flatten, Conv2D, Dropout, Input, \
    Reshape
from tensorflow.python.keras import optimizers, backend
from tensorflow.python.keras import activations
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import initializers, regularizers

from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import TensorBoard

from TetrisEnv import Tetris
import tkinter as tk
import matplotlib.pyplot as plt
import time

EPISODES = 50000


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    np.random.seed(1)

    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN

        self.discount_factor = 0.99
        self.LEARNING_RATE = 0.001  # best is 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_ddqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (4, 4), strides=(2, 2), input_shape=(24, 10, 1),  # batch_size=64,
                         kernel_initializer=initializers.glorot_uniform(), activation=activations.relu,
                         kernel_regularizer=regularizers.l2(0.01)))  # kernel initialize weights

        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activations.relu))

        model.add(Conv2D(128, (2, 2), strides=(1, 1), activation=activations.relu))

        # model.add(Dropout(0.5))
        model.add(Flatten())

        # model.add(Dense(512, input_dim=self.state_size, activation=activations.linear))  # autograd,PLRelu,RMS Prob
        # # model.add(LeakyReLU(alpha=0.3))
        # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        # # model.add(LeakyReLU(alpha=0.3))
        # model.add(Dense(256, activation=activations.linear))
        # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        # model.add(Dense(128, activation=activations.linear))
        # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))
        # model.add(Dense(64, activation=activations.linear))
        # model.add(BatchNormalization(momentum=0.99, epsilon=0.001))

        model.add(Dense(self.action_size, activation=activations.softmax))

        model.compile(loss=losses.categorical_crossentropy,  # loss='mse' losses.categorical_crossentropy
                      optimizer=optimizers.RMSprop(lr=self.LEARNING_RATE))  # RMSprob,Adam,Nadam
        self.tensorBoard = TensorBoard('./logs/RLAgent', histogram_freq=0,
                                       write_graph=True, write_images=True)
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, 24, 10, 1))

        update_target = np.zeros((batch_size, 24, 10, 1))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


def showDiagram():
    # x, y = fargs
    ax.clear()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Used(%)')
    # ax.legend(loc=2)

    # plt.legend(loc=2)
    # line1, = ax.plot(xs, ys, '-b', label='mean reward', linewidth=.5)
    line1, = ax.plot(xs, ys, 'o', color='blue')
    # canvas.show()
    plt.pause(.000001)


xs = []
ys = []

# ready diagram
plt.interactive(True)
f = plt.figure(figsize=(7, 5), dpi=100)
ax = f.add_subplot(111)

if __name__ == "__main__":

    root = tk.Tk()
    root.geometry('+%d+%d' % (800, 10))
    env = Tetris(root, render=True)

    state_size, action_size = env.getStateActionSize()
    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        losed = False
        score = 0
        state = env.reset()
        # state = np.zeros([64, 24, 10, 1])
        # state = np.reshape(state, [1, state_size])
        while not losed:
            # plt.pause(.0001)

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            (next_state), reward, losed = env.step(action)

            time.sleep(0.001)
            # next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            # print(reward)
            reward = reward if not losed else reward - 5

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, losed)
            # every time step do the training
            agent.train_model()
            state = next_state

            if losed:
                # every episode update the target model to be same with model
                agent.update_target_model()
                episodes.append(e)

                score = np.sum(state) * .5
                scores.append(score)

                print("episode:{}/{}  used:{:.4}% e: {:.2}".format(e, EPISODES, score, agent.epsilon))

                xs.append(e)
                ys.append(score)
                showDiagram()
                # pylab.plot(xs, ys, 'b')

                pylab.savefig("./ddqn_output/save_graph/tetris_ddqn.png")

                # if the mean of scores of last 10 episode is bigger than 90
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 90:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            agent.model.save_weights("./ddqn_output/save_model/tetris_ddqn.h5")
