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
        [(x - 1) % shape[0], y], [(x + 1) % shape[0], y],
        [x, (y - 1) % shape[1]], [x, (y + 1) % shape[1]],
        # [(x + 1) % shape[0], (y + 1) % shape[1]], [(x - 1) % shape[0], (y - 1) % shape[1]],
        # [(x + 1) % shape[0], (y - 1) % shape[1]], [(x - 1) % shape[0], (y + 1) % shape[1]]
    ]
    return neighbours


def count_force(array, x, y):
    counts = [0, 0, 0, 0]
    for nx in range(array.shape[0]):
        for ny in range(array.shape[1]):
            if array[nx, ny] > 0:
                if x <= array.shape[0] // 2:
                    if x < nx < x + array.shape[0] // 2:
                        # counts[1] += (nx - x) * array[nx, ny]
                        counts[1] += array[nx, ny] / (nx - x) ** 2
                    elif x - nx != 0:
                        # counts[0] += (x - nx) % array.shape[0] * array[nx, ny]
                        counts[0] += array[nx, ny] / ((x - nx) % array.shape[0]) ** 2
                else:
                    if x - array.shape[0] // 2 < nx < x:
                        # counts[0] += (x - nx) * array[nx, ny]
                        counts[0] += array[nx, ny] / (x - nx) ** 2
                    elif nx - x != 0:
                        # counts[1] += (nx - x) % array.shape[0] * array[nx, ny]
                        counts[1] += array[nx, ny] / ((nx - x) % array.shape[0]) ** 2
                if y <= array.shape[1] // 2:
                    if y < ny < y + array.shape[1] // 2:
                        # counts[3] += (ny - y) * array[nx, ny]
                        counts[3] += array[nx, ny] / (ny - y) ** 2
                    elif y - ny != 0:
                        # counts[2] += (y - ny) % array.shape[1] * array[nx, ny]
                        counts[2] += array[nx, ny] / ((y - ny) % array.shape[1]) ** 2
                else:
                    if y - array.shape[1] // 2 < ny < y:
                        counts[2] += array[nx, ny] / (y - ny) ** 2
                    elif ny - y != 0:
                        counts[3] += array[nx, ny] / ((ny - y) % array.shape[1]) ** 2
    # for i in range(len(counts)):
    #     if counts[i] > 100:
    #         counts[i] = 100
    return counts


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


size = 16

b = 2
db = .005
b_Max = 2
N_iter = 100
j = np.log10(size * size)

b = b / (10 ** j)
print(b)
regime = 0  # 0 - time evolution from rand state
mode = 0  # 0 - classical

# state = random_state_alpha(size, size, 0.9)


state  = dead_state(size,size)
# state[size//2-1:size//2+1,size//2-1:size//2+1 ] =100000
state[size//2,size//2 ] =10000
state[0,0 ] =10000
state[size//4,size//4] = 10
# state[4:16,4:7] =3
# state[4:7,4:16] =3
# state[4:16,13:16] =3
# state[13:16,4:16] =3
# for i in range(size):
#     state[i,i] = 1
# state[10:15, 10:50] = -1
# state[20:25, 10:50] = -1


# state[1, 1] = -1


def new_state(array):
    for k in range(N_iter):
        x = np.random.randint(array.shape[0])
        y = np.random.randint(array.shape[1])
        cell = array[x][y]
        if cell > 0:
            neighbours = get_neighbours(array.shape, x, y)
            probabilities = count_force(array, x, y)
            neighbours.append([x, y])
            probabilities.append(array[x, y])
            probabilities = np.asarray(probabilities) * b
            # print(probabilities)
            probabilities = np.exp(probabilities)
            probabilities = np.true_divide(probabilities, np.sum(probabilities))
            # print(probabilities)
            target = np.random.choice(np.arange(0, len(probabilities), 1, dtype=int), p=probabilities)
            x_target, y_target = neighbours[target]
            # array[x, y], array[x_target, y_target] = array[x_target, y_target], array[x, y]
            array[x, y] -= 1
            array[x_target, y_target] += 1
    return array


fig, ax1 = plt.subplots(1, 1)

im = ax1.imshow(state, animated=True, vmin=min(np.min(state), -1), vmax=10, cmap='PuBu')

flag = 0
period = 0


def animate(i):
    global state
    global b
    global flag, period, b_Max

    state = new_state(state)

    im.set_array(state)

    return [im]


anim = animation.FuncAnimation(fig, animate, interval=100, blit=True)
plt.show()
