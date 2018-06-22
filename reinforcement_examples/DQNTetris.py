# Cartpole DQN


# this is for multiple gpu
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tkinter as tk
# class for diagram
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from tkinter import ttk

import numpy as np
from DQNAgent import DQNAgent
from TetrisEnv import Tetris
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from multiprocessing import Process, Pipe

# set parameters

root = tk.Tk()
root.geometry('+%d+%d' % (800, 10))
env = Tetris(root, True)

state_size, action_size = env.getStateActionSize()
batch_size = 32
n_episodes = 100000
output_dir = 'model_output/RLTetris'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# style.use('fivethirtyeight')

xs = []
ys = []

# ready diagram
plt.interactive(True)
f = plt.figure(figsize=(7, 5), dpi=100)
ax = f.add_subplot(111)

# canvas = FigureCanvasTkAgg(f, root)
# canvas.get_tk_widget().grid(row=5, column=1)

agent = DQNAgent(state_size, action_size)


def showDiagram(x, y):
    # x, y = fargs
    ax.clear()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legend(loc=2)

    # plt.legend(loc=2)
    line1, = ax.plot(xs, ys, '-b', label='reward', linewidth=1)
    # canvas.show()
    plt.pause(0.001)


for e in range(n_episodes):
    state = env.reset()
    # print(state)
    # state = np.reshape(state, [1, state_size])
    losed = False
    while (not losed):
        action = agent.act(state)
        (next_state), reward, losed = env.step(action)
        reward = reward + 5 if not losed else reward - 10
        # next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, losed)
        state = next_state
        # empty = state[0][3]

        if losed:
            print("episode:{}/{}, reward:{}, e: {:.2}".format(e, n_episodes, reward, agent.EPSILON))
            xs.append(e)
            ys.append(reward)
            showDiagram(e, reward)
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # if e % 50 == 0:
        #     agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")

plt.savefig('fig.png', bbox_inches='tight')
agent.save('model.h5')
