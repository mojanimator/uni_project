import matplotlib.pyplot as plt
import random
import numpy as np

plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
xs = []
ys = []


def showDiagram(x, y):
    # x, y = fargs
    ax1.clear()
    # plt.clf()
    xs.append(x)
    ys.append(y)
    ax1.plot(xs, ys)

    plt.show(block=False)
    plt.pause(1)
    print(xs, ys)


for i in range(20):
    showDiagram(i, random.randint(1, 30))
