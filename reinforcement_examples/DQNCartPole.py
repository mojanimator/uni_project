# Cartpole DQN


# this is for multiple gpu
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import random
import gym
from time import sleep
import numpy as np
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from multiprocessing import Process, Pipe

# set parameters


env = gym.make('CartPole-v0')

state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n  # 2
batch_size = 32
n_episodes = 100
output_dir = 'model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

style.use('fivethirtyeight')

# plt.ion()
plt.interactive(True)
ax1 = plt.figure().add_subplot(1, 1, 1)
plt.title('Train')
plt.xlabel('episode')
plt.ylabel('score')

xs = []
ys = []

agent = DQNAgent(state_size, action_size)

done = False


def showDiagram(x, y):
    # x, y = fargs
    ax1.clear()
    # plt.clf()
    xs.append(x)
    ys.append(y)
    ax1.plot(xs, ys)
    plt.show()
    plt.pause(0.001)


for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    # print('state', state)
    for time in range(5000):
        # env.render()
        # print('episode:', e, 'time:', time)

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)  # if not lose reward is 1
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print("episode:{}/{}, score:{}, e: {:.2}".format(e, n_episodes, time, agent.EPSILON))
            # xs.append(e)
            # ys.append(time)
            showDiagram(e, time)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

plt.pause(50)
