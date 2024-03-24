import numpy as np
import matplotlib.pyplot as plt
from network import Network
import matplotlib.colors


# Class this (or simplify)
def line(network: Network):
    def graph_data(
        x_range: tuple[int, int],
        y_range: tuple[int, int],
        points=200,
        neutral="blue",
        negative="red",
    ):
        x = np.random.uniform(*x_range, points)
        y = np.random.uniform(*y_range, points)

        # colors = np.where(
        #     (y_range[1] + y_range[0] - y) * (x_range[1] + x_range[0])
        #     < x * (y_range[1] + y_range[0]),
        #     neutral,
        #     negative,
        # )

        colors = np.where(
            y > x - 0.2,
            neutral,
            negative,
        )

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(x, y, c=colors)
        ax.set_ylabel("Bpm")
        ax.set_xlabel("Age")
        ax.set_yticklabels(np.linspace(50, 100, 6))
        ax.set_xticklabels(np.linspace(0, 80, 6))

        colors = y > x - 0.2

        # colors = (y_range[1] + y_range[0] - y) * (x_range[1] + x_range[0]) < x * (
        #     y_range[1] + y_range[0]
        # )

        return ax, x, y, colors, fig

    # x_range = (0, 120)
    # y_range = (0, 120)
    # x_range = (0, 80)
    # y_range = (50, 100)
    x_range = (0, 1)
    y_range = (0, 1)
    ax, x, y, colors, fig = graph_data(x_range, y_range)

    points = np.column_stack((x, y, colors.astype(int)))
    xs, ys = np.meshgrid(np.linspace(*x_range), np.linspace(*y_range))
    pixels = np.stack((xs.T, ys.T), axis=2)

    def accuracy():
        total = 0
        for point in points:
            px, py, value = point
            output = network.calculate_output((px, py))
            total += int(output > 0) == value

        return total / len(points)

    img = network.calculate_outputs(pixels)

    custom_cmap = matplotlib.colors.ListedColormap(["#ADD8E6", "white"])

    img = plt.imshow(
        img,
        cmap=custom_cmap,
        extent=[*x_range, *y_range],
        aspect="equal",
        interpolation="gaussian",
        origin="lower",
    )

    def update():
        img.set_array(network.calculate_outputs(pixels))

    return update, accuracy, np.column_stack((x, y)), np.reshape(colors, (-1, 1)), fig
