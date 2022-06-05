import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random


def dead_state(height, width):
    array = np.zeros((height, width), dtype=int)
    return array


def random_state(width, height, mx):
    array = np.random.randint(0, high=mx, size=(width, height), dtype=int)
    return array


def random_state_alpha(width, height, mx):
    array = np.random.choice([0, 1], p=[mx, 1 - mx], size=(width, height))
    return array


def delta_state(width, height, mx):
    array = np.zeros((height, width), dtype=int)
    array[height // 2, width // 2] = mx
    return array


def get_neighbours(shape, x, y):
    neighbours = [
        [(x + 1) % shape[0], y], [(x - 1) % shape[0], y],
        [x, (y - 1) % shape[1]], [x, (y + 1) % shape[1]],
        [(x + 1) % shape[0], (y + 1) % shape[1]], [(x - 1) % shape[0], (y - 1) % shape[1]],
        [(x + 1) % shape[0], (y - 1) % shape[1]], [(x - 1) % shape[0], (y + 1) % shape[1]]
    ]
    return neighbours


def new_params_b(b):
    global db
    global b_Max
    b = b + db
    if b >= b_Max:
        db = -db
    if b <= -3:
        db = -db
    return b


def cooling_b(b):
    global db
    return b + db


size = 64

b = 0.1
db = .005
b_Max = 2
N_iter = 1000

regime = 1  # 0 - time evolution from rand state , 1 - hysteresis, 2 - cooling
mode = 0  # 0 - classical, 1 - boson, 2 - merging

state = random_state_alpha(size, size, 0.9)


# state[10:15, 10:50] = -1
# state[20:25, 10:50] = -1


# state[1, 1] = -1


def new_state(array):
    neig_pos = 0
    iter_pos = 1
    for k in range(N_iter):
        x = np.random.randint(array.shape[0])
        y = np.random.randint(array.shape[1])
        cell = array[x][y]
        if cell == 1:
            iter_pos += 1
            neighbours = get_neighbours(array.shape, x, y)
            closest = []
            probabilities = []
            nonzero_neigh = 0
            for xn, yn in neighbours:
                closest.append(xn + array.shape[0] * yn)
                if array[xn, yn] >= 1:
                    probabilities.append(0)
                    nonzero_neigh += 1
                    continue
                if array[xn, yn] < 0:
                    closest.pop(-1)
                    continue
                aux_neighbours = get_neighbours(array.shape, xn, yn)
                buf = 0
                for aux_x, aux_y in aux_neighbours:
                    if array[aux_x, aux_y] == 1:
                        buf += 1
                probabilities.append(buf)

            closest.append(x + array.shape[0] * y)
            probabilities.append(nonzero_neigh * array[x, y])
            neig_pos += nonzero_neigh
            probabilities = np.asarray(probabilities) * b
            probabilities = np.exp(probabilities)
            probabilities = np.true_divide(probabilities, np.sum(probabilities))
            target = np.random.choice(closest, p=probabilities)
            x_target = target % array.shape[0]
            y_target = target // array.shape[0]
            array[x, y], array[x_target, y_target] = array[x_target, y_target], array[x, y]

    return array, [neig_pos / iter_pos]


def new_state_coupling(array):
    neig_pos = 0
    iter_pos = 1
    stMax = 1
    for k in range(N_iter):
        x = np.random.randint(array.shape[0])
        y = np.random.randint(array.shape[1])
        cell = array[x][y]
        if cell >= 1:
            iter_pos += 1
            neighbours = get_neighbours(array.shape, x, y)
            closest = []
            probabilities = []
            nonzero_neigh = 0
            for xn, yn in neighbours:
                closest.append(xn + array.shape[0] * yn)
                if array[xn, yn] >= 1:
                    probabilities.append(0)
                    nonzero_neigh += 1
                    continue
                if array[xn, yn] < 0:
                    closest.pop(-1)
                    continue
                aux_neighbours = get_neighbours(array.shape, xn, yn)
                buf = 0
                for aux_x, aux_y in aux_neighbours:
                    if array[aux_x, aux_y] >= 1:
                        buf += 1
                probabilities.append(buf)
            if nonzero_neigh != len(neighbours):
                closest.append(x + array.shape[0] * y)
                probabilities.append(nonzero_neigh)
            neig_pos += nonzero_neigh
            probabilities = np.asarray(probabilities) * b
            probabilities = np.exp(probabilities)
            # print(probabilities)
            probabilities = np.true_divide(probabilities, np.sum(probabilities))
            target = np.random.choice(closest, p=probabilities)
            x_target = target % array.shape[0]
            y_target = target // array.shape[0]
            # array[x, y], array[x_target, y_target] = array[x_target, y_target], array[x, y]
            array[x, y] -= 1
            array[x_target, y_target] += 1
            # if array[x_target, y_target] > stMax:
            #     stMax = array[x_target, y_target]
            #     print(stMax)
    return array, [neig_pos / iter_pos / len(get_neighbours(array.shape, 0, 0))]


def new_state_merging(array):
    neig_pos = 0
    iter_pos = 1
    for k in range(N_iter):
        x = np.random.randint(array.shape[0])
        y = np.random.randint(array.shape[1])
        cell = array[x][y]
        if cell >= 1:
            iter_pos += 1
            neighbours = get_neighbours(array.shape, x, y)
            closest = []
            probabilities = []
            nonzero_neigh = 0
            for xn, yn in neighbours:
                closest.append(xn + array.shape[0] * yn)

                if array[xn, yn] < 0:
                    closest.pop(-1)
                    continue
                aux_neighbours = get_neighbours(array.shape, xn, yn)
                buf = 0
                for aux_x, aux_y in aux_neighbours:
                    if array[aux_x, aux_y] >= 1:
                        buf += 1
                if array[xn, yn] >= 1:
                    buf -= 1
                    neig_pos += 1
                probabilities.append(buf)
            if nonzero_neigh != len(neighbours):
                closest.append(x + array.shape[0] * y)
                probabilities.append(nonzero_neigh)
            neig_pos += nonzero_neigh
            probabilities = np.asarray(probabilities) * b
            probabilities = np.exp(probabilities)
            probabilities = np.true_divide(probabilities, np.sum(probabilities))
            target = np.random.choice(closest, p=probabilities)
            x_target = target % array.shape[0]
            y_target = target // array.shape[0]

            array[x, y] -= 1
            array[x_target, y_target] += 1

    return array, [neig_pos / iter_pos / len(get_neighbours(array.shape, 0, 0))]


fig, (ax1, ax2) = plt.subplots(2, 1)

StateMax = np.max(state)
if mode != 0:
    im = ax1.imshow(state, animated=True, vmin=min(np.min(state), -1), vmax=10, cmap='plasma')
if mode == 0:
    im = ax1.imshow(state, animated=True, vmin=min(np.min(state), -1), vmax=1, cmap='PuBu')
line1, = ax2.plot([], [], lw=2, label="Mean num of neighbours")
if mode != 0:
    line2, = ax2.plot([], [], lw=2, label="Num of cells")

ax2.set_ylim(-1, 1)
ax2.set_xlim(0, size)

ax2.set_xlabel('step')

if regime == 1 or regime == 2:
    ax2.set_xlabel('b')
    ax2.set_ylim(-1, 1)
    ax2.set_xlim(-b_Max * 1.1, b_Max * 1.1)

leg = ax2.legend()
ax2.grid()
xdata, ydata1, ydata2 = [], [], []

flag = 0
period = 0


def animate(i):
    global state
    global b
    global flag, period, b_Max

    x_axes = i

    if regime == 1:
        b = new_params_b(b)
        x_axes = b
    if regime == 2:
        b = cooling_b(b)
        x_axes = b

    if mode == 0:
        state, counts = new_state(state)
    if mode == 1:
        state, counts = new_state_coupling(state)
    if mode == 2:
        state, counts = new_state_merging(state)

    xdata.append(x_axes)
    ydata1.append(counts)
    if mode != 0:
        ydata2.append(np.count_nonzero(state) / (np.size(state)))
    im.set_array(state)

    xmin, xmax = ax2.get_xlim()
    ymin, ymax = ax2.get_ylim()

    if regime == 1 and b >= b_Max:
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
    if regime == 0 and mode == 1:
        line2.set_data(xdata, ydata2)

    if mode != 0:
        return [line1, line2, im]
    return [line1, im]


anim = animation.FuncAnimation(fig, animate, interval=10, blit=True)
plt.show()
