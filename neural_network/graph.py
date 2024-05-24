import numpy as np
import matplotlib.pyplot as plt
from network import Network
import matplotlib.colors
import itertools


def graph(network: Network, equation):
    x_range = (0, 1)
    y_range = (0, 1)
    size = 36
    points = 256
    neutral = "blue"
    negative = "red"
    square_area = 200

    lin = np.linspace(0, 1, size)

    xs, ys = np.meshgrid(lin, lin)

    pixels = np.array([*itertools.product(lin, repeat=2)])

    shaded_area = network.calculate_outputs_graph(pixels)

    fig = plt.figure(num="Graph") #figsize=(12, 12)
    ax = fig.add_subplot(111)

    graph = ax.scatter(xs, ys, shaded_area * square_area, c="#ADD8E6", marker="s")

    x = np.random.uniform(*x_range, points)
    y = np.random.uniform(*y_range, points)

    network_shade = equation(x, y)

    labels = np.column_stack((equation(y, x), (1 - equation(y, x))))

    # print(labels, network_shade)
    # labels = np.rot90(np.column_stack(((1 - network_shade), network_shade)), k=2)

    # j = network_shade.reshape(16, -1)
    # print(j.shape)
    # ax2.imshow(np.array(list(zip(x, y))), cmap=j)
    # j = network_shade.reshape(16, -1)
    # ax2.imshow(shaded_area.reshape(6, -1))
    # ax.scatter(xs, ys, equation(xs, ys).astype(int) * square_area, c="purple", marker="s")
    # colors = np.where(network_shade, neutral, negative)
    # scatter = ax.scatter(x, y, c=colors)

    ax.plot(
        x[network_shade],
        y[network_shade],
        "bo",
        x[~network_shade],
        y[~network_shade],
        "ro",
    )

    # plt.legend(["Do not have disease", "Have disease"])

    ax.set_ylabel("Bpm")
    ax.set_xlabel("Age")
    ax.set_yticklabels(np.round(np.linspace(50, 100, 7)))
    ax.set_xticklabels(np.round(np.linspace(0, 80, 7)))
    # handles, labels = scatter.legend_elements()
    # legend = ax.legend(handles=handles, labels=["Do not have disease", "Have disease"], title="Data Points")

    def update():
        img = network.calculate_outputs_graph(pixels)
        graph.set_sizes(img * square_area)

    # TODO accuracy func

    return update, np.array(list(zip(x, y))), labels
