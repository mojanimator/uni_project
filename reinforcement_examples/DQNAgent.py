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

from TetrisEnv_w_test import Tetris
import tkinter as tk
import matplotlib.pyplot as plt
import pylab
import time
import os

if (not os.path.exists('./dqn_output')):
    os.mkdir('./dqn_output')
    os.mkdir('./dqn_output/save_model')
    os.mkdir('./dqn_output/save_graph')


# play many episodes and remember some information from episodes randomly in memory
# and replay them
class DQNAgent:
    def __init__(self, state_size, action_size):
        np.random.seed(1)
        self.state_size = state_size
        self.action_size = action_size
        self.MEMORY_LEN = 5000
        self.TRAIN_START = 1000
        self.memory = deque(maxlen=self.MEMORY_LEN)
        self.GAMMA = 0.9  # degree of feature effect (discount factor)
        self.EPSILON = 1.0  # exploration rate (in beginning)
        self.EPSILON_DECAY = 0.9999  # slowly shift exploration to exploitation
        self.EPSILON_DECAY = 0.00002  # slowly shift exploration to exploitation
        self.EPSILON_MIN = 0.01
        self.LEARNING_RATE = 0.001  # 0.001  # gradient descent rate 0.001
        self.EPOCHS = 1
        self.BATCH_SIZE = 32
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        # input shape (samples, channels, rows, cols)
        model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(24, 10, 1),
                         kernel_initializer=initializers.glorot_uniform(), activation=activations.relu,
                         kernel_regularizer=regularizers.l2(0.01)))  # kernel initialize weights

        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activations.relu))

        model.add(Conv2D(128, (3, 3), strides=(1, 1), activation=activations.relu))

        # model.add(Dropout(0.5))
        model.add(Flatten())

        model.add(Dense(self.action_size, activation=activations.softmax))

        model.compile(loss=losses.categorical_crossentropy,  # loss='mse' losses.categorical_crossentropy
                      optimizer=optimizers.Nadam(lr=self.LEARNING_RATE))  # RMSProb,Adam
        self.tensorBoard = TensorBoard('./logs/RLAgent', histogram_freq=0,
                                       write_graph=True, write_images=True)
        model.summary()
        return model

    def remember(self, state, action, reward, next_state, done):
        # if len(self.memory) >= self.MEMORY_LEN:
        #     self.memory.popleft()
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        # print('act_value',(act_values[0]))
        return np.argmax(act_values[0])

    def train_model(self):
        if len(self.memory) < self.TRAIN_START:
            return

        miniBatch = random.sample(self.memory, self.BATCH_SIZE)
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
            self.EPSILON -= self.EPSILON_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def showDiagram():
    # x, y = fargs
    ax.clear()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Used(%)')
    # ax.legend(loc=2)

    plt.legend(loc='best')
    # line1, = ax.plot(xs, ys, '-b', label='mean reward', linewidth=.5)
    line1, = ax.plot(xs, ys, label='mean reward', linewidth=.5)
    # line1, = ax.plot(xs, ys, 'o', color='blue')
    # canvas.show()
    plt.pause(.000001)


xs = []
ys = []

# ready diagram
plt.interactive(True)
f = plt.figure(figsize=(15, 9), dpi=100)
ax = f.add_subplot(111)


def simplify(r):
    r = np.array(r)
    res = np.mean(r.reshape(-1, 20), axis=1)

    return range(1, len(r), 20), res


if __name__ == "__main__":

    EPISODES = 2200
    root = tk.Tk()
    root.geometry('+%d+%d' % (800, 10))
    env = Tetris(root, render=True)

    state_size, action_size = env.getStateActionSize()
    # agent = DoubleDQNAgent(state_size, action_size)

    weights = [[.6, 3.1, .6, .1], [.1, 4.0, .6, .3],
               [3.2, 3.9, 1.9, 2.1], [0.71, 2.38, 1.32, 0.1], [2.23, 1.77, 0.87, 0.55], [2.7, 1.17, 1.17, 0.4],
               [4.0, 2.0, 2.0, 0.0], [1.225, .85, .85, .01], [.47, 1.2, 0.1, .01], [1.4, 1.13, 3.45, 0.1],
               [1.6, 1.6, .47, .01], [1.6, 1.6, .47, .1], [1.6, 1.6, 1.6, .01], [1.8, 1.0, 1.24, .01],
               [.1, 2.0, .1, .67],
               [.48, .86, .86, .01],
               ]

    # testRange=np.linspace(0,1.5,num=5)
    #
    # [ weights.append([i,j,k,l]) for i in np.linspace(0.5,1.5,num=5) for j in testRange for k in testRange for l in testRange]
    # test = np.linspace(.1, 2, num=11)
    # print(test)
    # [weights.append([random.choice(test), random.choice(test), random.choice(test), 0.01]) for _ in range(60)]
    results = []
    ep = list(range(EPISODES))

    for idx, w in enumerate(weights):
        agent = DQNAgent(state_size, action_size)
        scores = []
        start_time = time.time()
        for e in range(EPISODES):
            losed = False
            e_score = 0
            state = env.reset(weights=w)
            # state = np.zeros([64, 24, 10, 1])
            # state = np.reshape(state, [1, state_size])
            while not losed:
                # plt.pause(.0001)

                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                (next_state), reward, losed = env.step(action)

                # next_state = np.reshape(next_state, [1, state_size])
                # if an action make the episode end, then gives penalty of -100
                # print(reward)
                reward = reward if not losed else reward - 100

                # save the sample <s, a, r, s'> to the replay memory
                agent.remember(state, action, reward, next_state, losed)
                # every time step do the training
                agent.train_model()
                state = next_state

                if losed:
                    # every episode update the target model to be same with model

                    # episodes.append(e)

                    e_score = np.sum(state) * .5
                    scores.append(e_score)

                    print("test:{} episode:{}/{}  used:{:.4}% e: {:.4}".format(idx + 1, e, EPISODES, e_score,
                                                                               agent.EPSILON))

                    # xs.append(e)
                    # ys.append(score)
                    # showDiagram()
                    # pylab.plot(xs, ys, 'b')

                    # pylab.savefig("./ddqn_output/save_graph/tetris_ddqn.png")
                    plt.pause(0.00001)
                    # if the mean of scores of last 10 episode is bigger than 90
                    # stop training
                    # if np.mean(scores[-min(10, len(scores)):]) > 90:
                    #     sys.exit()

            # if e % 50 == 0:
            #     agent.model.save_weights("/dqn_output/save_model/tetris_dqn.h5")

        # collect results for every weights
        results.append(scores)
        # time
        print("test:{}  runtime:{:.4} minute".format(idx + 1, (time.time() - start_time) / 60))
        # save the model
        agent.model.save_weights("./dqn_output/save_model/tetris_dqn_"
                                 + str(weights[idx][0]) + "_" + str(weights[idx][1]) + "_" +
                                 str(weights[idx][2]) + "_" + str(weights[idx][3]) + ".h5")
        l = idx + 1
        offset = 2
        if (l % offset == 0):
            f.clf()
            plt.xlabel('Episode')
            plt.ylabel('Used(%)')

            for id, r in enumerate(results[l - offset:l]):
                index = l - offset + id
                ep, r = simplify(r)  # mean every 20 steps
                plt.plot(ep, r, label=" weight {:.4},{:.4},{:.4},{:.4}"
                         .format(weights[index][0], weights[index][1], weights[index][2], weights[index][3]))

            plt.legend(loc=3)
            # save file
            file = './dqn_output/save_graph/tetris_dqn_' + str(l - offset) + '_' + str(l) + '.png'
            pylab.savefig(file)
