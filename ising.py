import math
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


def dead_state(height, width):
    array = np.full((height, width), dtype=float)
    return array


def random_state(width, height):
    array = np.random.choice([-1, 1], width * height, p=[0.5, 0.5]).reshape((width, height))
    return array


def delta_state(width, height):
    array = np.ones((height, width), dtype=int)
    array[height // 2, width // 2] = -10
    return array


def get_neighbours(shape, x, y):
    neighbours = [[(x + 1) % shape[0], y], [(x - 1) % shape[0], y], [x, (y - 1) % shape[1]],
                  [x, (y + 1) % shape[1]]]
    return neighbours

def new_params_H(H):
    global dH
    global H_Max
    H = H + dH
    if H >= H_Max:
        dH = -dH
    if H <= - H_Max:
        dH = -dH
    return H


def new_params_b(b):
    global db
    global b_Max
    b = b + db
    if b >= b_Max:
        db = -db
    if b <= 0:
        db = -db
    return b


def cooling_b(b):
    global db
    return b + db


size = 150

# 60 , 75 i - for phase transition
# it means +3, +3.3
# kTс = 2.27J
b = 3.3
H = 0.0
db = .01
dH = 2.
H_Max = 50
b_Max = 3.7
N_iter = 10000
temp = 1
J = 10.
regime = 0  # 0 - time evolution from rand state , 1 - hysteresis, 2 - evolution with temperature, 3 - cooling

state = random_state(size, size)


def new_state(array):
    for k in range(N_iter):
        x = np.random.randint(array.shape[0])
        y = np.random.randint(array.shape[1])
        cell = array[x][y]
        array[x][y] = -cell
        p = H
        neighbours = get_neighbours(array.shape, x, y)
        for xn, yn in neighbours:
            p += array[xn][yn]

        E_1 = -J * cell * p - H * cell
        E_2 = J * cell * p + H * cell
        if E_2 - E_1 <= 0:
            pass
        else:
            if random.uniform(0, 1) <= np.exp((E_1 - E_2) * b):
                pass
            else:
                array[x][y] = cell

    return array, np.sum(array) / (size * size)


fig, (ax1, ax2) = plt.subplots(2, 1)

im = ax1.imshow(state, animated=True, vmin=-1, vmax=1)
line1, = ax2.plot([], [], lw=2, label="M")

ax2.set_ylim(-1, 1)
ax2.set_xlim(0, size)

ax2.set_xlabel('step')
if regime == 1:
    ax2.set_xlabel('H')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-H_Max * 1.1, H_Max * 1.1)
if regime == 2 or regime == 3:
    ax2.set_xlabel('b')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(0, b_Max * 1.1)

leg = ax2.legend()
ax2.grid()
xdata, ydata1 = [], []

flag = 0
period = 0


def animate(i):
    global state
    global b, H
    global flag, H_Max, period, b_Max

    x_axes = i
    if regime == 1:
        H = new_params_H(H)
        x_axes = H
    if regime == 2:
        b = new_params_b(b)
        x_axes = b
    if regime == 3:
        b = cooling_b(b)
        x_axes = b

    state, counts = new_state(state)

    xdata.append(x_axes)
    ydata1.append(counts)

    im.set_array(state)

    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()

    if regime == 1 and H >= H_Max:
        if flag == 0:
            period = len(xdata)
        if flag == 1:
            period = len(xdata) - period
        flag += 1

        if flag == 4:
            del xdata[:-period]
            del ydata1[:-period]
            flag = 0
    if regime == 2 and b >= b_Max:
        if flag == 0:
            period = len(xdata)
        if flag == 1:
            period = len(xdata) - period
        flag += 1

        if flag == 4:
            del xdata[:-period]
            del ydata1[:-period]
            flag = 0

    if x_axes > xmax:
        ax2.set_xlim(xmin, 2 * xmax)
        ax2.figure.canvas.draw()
    if counts > ymax:
        ax2.set_ylim(ymin, 1.2 * ymax)
        ax2.figure.canvas.draw()

    line1.set_data(xdata, ydata1)

    return [line1, im]


anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)
plt.show()
