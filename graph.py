import numpy as np
import matplotlib.pyplot as plt
from network import Network
import matplotlib.colors
import itertools


def graph(network: Network, equation):
    x_range = (0, 1)
    y_range = (0, 1)
    size = 40
    points = 200
    neutral = "blue"
    negative = "red"
    square_area = 150

    lin = np.linspace(0, 1, size)

    xs, ys = np.meshgrid(lin, lin)

    pixels = np.array([*itertools.product(lin, repeat=2)])

    shaded_area = network.calculate_outputs_graph(pixels)

    fig = plt.figure(num="Graph")
    ax = fig.add_subplot(111)

    graph = ax.scatter(xs, ys, shaded_area * square_area, c="#ADD8E6", marker="s")

    x = np.random.uniform(*x_range, points)
    y = np.random.uniform(*y_range, points)

    network_shade = equation(x, y)

    labels = np.column_stack((1 - network_shade,network_shade))
    
    colors = np.where(
        network_shade,
        negative,
        neutral
    )
    
    #ax.scatter(xs, ys, equation(xs, ys).astype(int) * square_area, c="purple", marker="s")

    ax.scatter(x, y, c=colors)

    ax.set_ylabel("Bpm")
    ax.set_xlabel("Age")
    ax.set_yticklabels(np.linspace(50, 100, 6))
    ax.set_xticklabels(np.linspace(0, 80, 6))

    def update():
        img = network.calculate_outputs_graph(pixels)
        graph.set_sizes(img * square_area)

    # TODO accuracy func

    return update, np.array(list(zip(x,y))), labels
