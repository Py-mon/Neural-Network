import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """Shrinks a value between 0 and 1."""
    return 1 / (1 + np.exp(-x))


x = np.linspace(-7, 7, 100)
y = sigmoid(x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.set_ylim([-1, 2])

ax.grid()

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")

# Eliminate upper and right axes
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# Show ticks in the left and lower axes only
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

plt.title("Sigmoid Activation Function")
ax.plot(x, y, linewidth=3)

plt.show()

# https://stackoverflow.com/questions/31556446/how-to-draw-axis-in-the-middle-of-the-figure