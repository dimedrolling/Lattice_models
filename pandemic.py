import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


def dead_state(height, width):
    array = np.zeros((height, width), dtype=int)
    return array


def random_state(width, height,mx):
    array = np.random.randint(0, high=mx, size=(width, height), dtype=int)
    return array


def delta_state(width, height, mx):
    array = np.zeros((height, width), dtype=int)
    array[height // 2, width // 2] = mx
    return array


def get_neighbours(shape, x, y):
    neighbours = [
        # [(x + 1) % shape[0], y], [(x - 1) % shape[0], y],
        [x, (y - 1) % shape[1]], [x, (y + 1) % shape[1]],
        [(x + 1) % shape[0], (y + 1) % shape[1]], [(x - 1) % shape[0], (y - 1) % shape[1]],
        [(x + 1) % shape[0], (y - 1) % shape[1]], [(x - 1) % shape[0], (y + 1) % shape[1]]
    ]
    return neighbours


prob = 0.5
immune_step = 1
contagious_rate = 1
high_ill = 100
size = 256


def new_state(array):
    ill, immune, healthy = 0, 0, 0
    updated_array = array.copy()
    for x in range(0, array.shape[0]):
        for y in range(0, array.shape[1]):
            cell = array[x][y]
            if cell >= high_ill * contagious_rate:
                ill += 1
                neighbours = get_neighbours(array.shape, x, y)
                for xn, yn in neighbours:
                    p = random.uniform(0, 1)
                    if p < prob:
                        if updated_array[xn, yn] <= 0:
                            updated_array[xn, yn] = high_ill
            else:
                if cell > 0:
                    immune += 1

                if cell <= 0:
                    healthy += 1
            if cell > 0:
                updated_array[x, y] -= immune_step

    return updated_array, [ill, healthy, immune]


state = delta_state(size, size, high_ill)
# state[0,0] = high_ill

fig, (ax1, ax2) = plt.subplots(2, 1)

im = ax1.imshow(state, animated=True)
line1, = ax2.plot([], [], lw=2, label="contagious")
line2, = ax2.plot([], [], lw=2,label="healthy")
line3, = ax2.plot([], [], lw=2,label="immune")

ax2.set_ylim(0, size * size)
ax2.set_xlim(0, size)
leg = ax2.legend()
ax2.grid()
xdata, ydata1, ydata2, ydata3 = [], [], [], []


def animate(i):
    global state

    state, counts = new_state(state)
    xdata.append(i - 1)
    ydata1.append(counts[0])
    ydata2.append(counts[1])
    ydata3.append(counts[2])
    im.set_array(state)

    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()

    if i >= xmax:
        ax2.set_xlim(xmin, 2 * xmax)
        ax2.figure.canvas.draw()
    if max(counts) >= ymax:
        ax2.set_ylim(ymin, 1.2 * ymax)
        ax2.figure.canvas.draw()

    line1.set_data(xdata, ydata1)
    line2.set_data(xdata, ydata2)
    line3.set_data(xdata, ydata3)

    return [line1, line2, line3, im]


anim = animation.FuncAnimation(fig, animate, interval=1, blit=True)
plt.show()
