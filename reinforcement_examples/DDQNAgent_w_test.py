import sys
# import gym
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

from TetrisEnv_w_test import Tetris
import tkinter as tk
import matplotlib.pyplot as plt
import time
import os
import time

if (not os.path.exists('./ddqn_output')):
    os.mkdir('./ddqn_output')
    os.mkdir('./ddqn_output/save_model')
    os.mkdir('./ddqn_output/save_graph')


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    np.random.seed(1)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(24, 10, 1),
                         activation=activations.relu, ))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=activations.relu))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), activation=activations.relu))

        model.add(Flatten())

        model.add(Dense(self.action_size, activation=activations.softmax))
        model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Nadam(lr=self.LEARNING_RATE))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        # self.epsilon *= self.epsilon_decay

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
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        # F:\Ezafi\books\__ArshadBooks\0000finalProject\_project\reinforcement_examples
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0, )  # callbacks=[self.tensorBoard])  # tensorboard --logdir ./logs/DRLAgent

    def __init__(self, state_size, action_size):

        # self.render = False
        self.load_model = True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN

        self.discount_factor = 0.9
        self.LEARNING_RATE = 0.001  # best is 0.001
        self.epsilon = 0.001
        # self.epsilon_decay = 0.9999  # 0.9999 -> 2000 episodes   *=
        self.epsilon_decay = 0.00002  # 0.00002 -> 2500 episodes -=
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 64

        # create replay memory using deque
        self.memory = deque(maxlen=5000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # if self.load_model:
        #     self.model.load_weights("./save_model/tetris_ddqn.h5")


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

    EPISODES = 20000
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
    # test = np.linspace(.1, 5, num=20)
    # test = [0.5, 1.0, 2.0, 3.0, 4.0]
    # print(test)
    # [weights.append([random.choice(test), random.choice(test), random.choice(test), random.choice(test)])
    #  for _ in range(60)]
    results = []
    ep = list(range(EPISODES))

    for idx, w in enumerate(weights):
        agent = DoubleDQNAgent(state_size, action_size)
        agent.load_model = True
        if agent.load_model:
            agent.model.load_weights("./ddqn_output/save_model/tetris_ddqn_"
                                     + str(weights[idx][0]) + "_" + str(weights[idx][1]) + "_" +
                                     str(weights[idx][2]) + "_" + str(weights[idx][3]) + ".h5")
            print('model loaded !')

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
                agent.append_sample(state, action, reward, next_state, losed)
                # every time step do the training
                agent.train_model()
                state = next_state

                plt.pause(0.00001)
                if losed:
                    # every episode update the target model to be same with model
                    agent.update_target_model()
                    # episodes.append(e)

                    e_score = np.sum(state) * .5
                    scores.append(e_score)
                    print("test:{} episode:{}/{}  used:{:.4}% e: {:.4}".format(idx + 1, e, EPISODES, e_score,
                                                                               agent.epsilon))

                    # xs.append(e)
                    # ys.append(score)
                    # showDiagram()
                    # pylab.plot(xs, ys, 'b')

                    # pylab.savefig("./ddqn_output/save_graph/tetris_ddqn.png")

                    plt.pause(0.00001)

                    # if the mean of scores of last 10 episode is bigger than 70
                    # stop training
                    # if np.mean(scores[-min(10, len(scores)):]) > 90:
                    #     sys.exit()

        # collect results for every weights
        results.append(scores)
        # time
        print("test:{}  runtime:{:.4} minute".format(idx + 1, (time.time() - start_time) / 60))

        agent.model.save_weights("./ddqn_output/save_model/tetris_ddqn_"
                                 + str(weights[idx][0]) + "_" + str(weights[idx][1]) + "_" +
                                 str(weights[idx][2]) + "_" + str(weights[idx][3]) + ".h5")

        l = idx + 1
        offset = 8
        if (l % offset == 0):
            f.clf()
            plt.xlabel('Episode')
            plt.ylabel('Used(%)')

            for id, r in enumerate(results[l - offset:l]):  # results[l - offset:l]
                index = l - offset + id

                ep, r = simplify(r)  # mean every 20 steps
                plt.plot(ep, r, label=" weight {:.4},{:.4},{:.4},{:.4}"
                         .format(weights[index][0], weights[index][1], weights[index][2], weights[index][3]))

            plt.legend(loc=3)
            # save file
            file = './ddqn_output/save_graph/tetris_ddqn_' + str(l - offset) + '_' + str(l) + '.png'
            pylab.savefig(file)

# save file
# i=1
# file='./ddqn_output/save_graph/tetris_ddqn_'+str(i)+'.png'
# while(os.path.exists(file)):
#     i+=1
#     file = './ddqn_output/save_graph/tetris_ddqn_' + str(i) + '.png'
# pylab.savefig(  file)
# plot all results
