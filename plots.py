import numpy as np

from scipy.special import softmax
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import projections
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


import keras.api._v2.keras as keras


mnist = keras.datasets.mnist
(train, train_labels), (test, test_labels) = mnist.load_data()

train = keras.utils.normalize(train, axis=1)
test = keras.utils.normalize(test, axis=1)

# train = np.concatenate(train, axis=2)


def sigmoid(x):
    """Shrinks a value between 0 and 1."""
    return 1 / (1 + np.exp(-x))


def random_num(shape):
    """Random numbers between -1 and +1"""
    # return np.random.uniform(-1, 1, shape)
    if isinstance(shape, tuple):
        return np.random.randn(*shape)
    else:
        return np.random.randn(shape)


def expand_range(minimum: float, maximum: float, percent: float):
    """Expands a linspace range correctly with negatives by an percent."""
    return minimum - abs(minimum * percent), maximum + abs(maximum * percent)


class Network:
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.inputs = layers[0]
        self.biases = [
            random_num(y) for y in layers[1:]
        ]  # No biases in the input layer, hence the '[1:]'
        self.layers_setup = layers
        self.weights = [random_num((y, x)) for x, y in zip(layers[:-1], layers[1:])]

        self.loss_gradient_weights = [np.empty_like(x) for x in self.weights]
        self.loss_gradient_biases = [np.empty_like(x) for x in self.biases]

        self.variable1: tuple[list, int, int, int] = (self.weights, 0, 0, 0)
        self.variable2: tuple[list, int, int, int] = (self.weights, 1, 0, 0)

        self.path = []

    def get_variable(self, variable):
        return variable[0][variable[1]][variable[2:]]

    def null_biases(self):
        """nullifies the biases to 0"""
        self.biases = np.zeros_like(self.biases)

    def feed_forward(self, input):
        self.activations = []
        for b, w in zip(self.biases, self.weights):
            input = sigmoid(np.dot(w, input) + b)  # + b
            self.activations.append(input)
        self.layers = list(zip(self.biases, self.weights, self.activations))
        # self.activations = 1 / ( 1 + np.exp(np.dot(self.weights, input) + self.biases) )
        return input

    def __call__(self, inputs):
        return self.feed_forward(inputs)

    def loss(self, prediction, label):
        # have multiple output nodes
        # return (prediction - label) ** 2
        return np.sum((prediction - label) ** 2)

    def get_loss(self, input, labels):
        # loss = 0
        # for i, inputs in enumerate(input):
        #     output = self.feed_forward(inputs)
        #     label = np.zeros(10)
        #     label[labels[i]] = 1
        #     loss += self.loss(output, label)
        # print(loss)
        # return loss

        # outputs = np.array([self(input_vector) for input_vector in input])

        outputs = self(input)

        loss = self.loss(outputs, labels)

        return loss

    def expand_range(self, minimum, maximum, percent):
        """Expands a linspace range correctly with negatives by an percent."""
        return minimum - abs(minimum * percent), maximum + abs(maximum * percent)

    def calculate_losss_of_weights(
        self,
        input,
        output,
        points,
        range_,
    ):
        variable1s = np.linspace(-range_, range_, points)
        variable2s = np.linspace(-range_, range_, points)

        x, y, z = (
            np.empty(points**2),
            np.empty(points**2),
            np.empty((points, points)),
        )
        for i, variable1 in enumerate(variable1s):
            for j, variable2 in enumerate(variable2s):
                parameter, layer_index, *index = self.variable1
                parameter[layer_index][index] = variable1
                parameter, layer_index, *index = self.variable2
                parameter[layer_index][index] = variable2
                loss = self.get_loss(input, output)
                index = i * points + j
                x[index] = variable1
                y[index] = variable2
                z[i, j] = loss
        return x, y, z

    def plot_2d_gradient(self, input, labels, points=500, value=10):
        """Satisfies the xy = 0.5 equation.
        ```
            |
        ___  \\___
           \\
            |
            """
        x, y, z = self.calculate_losss_of_weights(
            input,
            labels,
            points,
            value,
        )

        print(x, y, z)

        # Plot Image
        plt.imshow(z, cmap=cm.jet, interpolation="bilinear")

        ax = plt.gca()
        ax.set_yticklabels(np.round(np.linspace(min(y), max(y), 10)))
        ax.set_xticklabels(np.round(np.linspace(min(x), max(x), 10)))

        ax.set_xlabel("weight1")
        ax.set_ylabel("weight2")

        plt.show()

    def plot_3d_gradient(self, input, labels, points=20, value=10, path=True):
        x, y, z = self.calculate_losss_of_weights(
            input,
            labels,
            points,
            value,
        )
        _, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection="3d"))

        # ax.set_zticklabels([f"10^{round(x)}" for x in ax.get_zticks()])

        # ax.plot_trisurf(x, y, np.log10(z), cmap=cm.jet, linewidth=0.1)
        ax.scatter(x, y, z)
        # ax.zaxis.set_minor_locator(ticker.LogLocator(base=10))
        # ax.zscale('symlog')
        # ax.set_zscale('symlog')
        ax.set_xlabel("weight1")
        ax.set_ylabel("weight2")
        ax.set_zlabel("loss")
        if path:
            ax.scatter(*np.transpose(self.path), color="r")

        plt.show()

    def randomize(self):
        self.biases = [
            random_num(y) for y in self.layers_setup[1:]
        ]  # No biases in the input layer, hence the '[1:]'

        self.weights = [
            random_num((y, x))
            for x, y in zip(self.layers_setup[:-1], self.layers_setup[1:])
        ]

    def learn(self, input, labels, learn_rate=1, plot=False):
        h = 0.000001
        original_loss = self.get_loss(input, labels)

        def add_path(loss):
            self.path += (
                (
                    self.get_variable(self.variable1),
                    self.get_variable(self.variable2),
                    loss,
                ),
            )

        for layer_i in range(self.num_layers - 1):
            for i, weights in enumerate(self.weights[layer_i]):
                for j, _ in enumerate(weights):
                    self.weights[layer_i, i, j] += h
                    loss = self.get_loss(input, labels)
                    change_in_loss = loss - original_loss

                    # if change_in_loss / h * learn_rate < 0.01:
                    #     print("R")
                    #     self.randomize()
                    #     self.null_biases()
                    #     continue

                    # add_path(loss)
                    self.weights[layer_i, i, j] -= h

                    self.loss_gradient_weights[layer_i][i][j] = change_in_loss / h

            for i, _ in enumerate(self.biases[layer_i]):
                self.biases[layer_i, i] += h
                change_in_loss = self.get_loss(input, labels) - original_loss
                self.biases[layer_i, i] -= h
                self.loss_gradient_biases[layer_i, i] = change_in_loss / h

        for i in range(self.num_layers - 1):
            self.biases[i] -= self.loss_gradient_biases[i] * learn_rate
            self.weights[i] -= self.loss_gradient_weights[i] * learn_rate

        loss = self.get_loss(input, labels)
        add_path(loss)
        print(loss)


# n = Network([784, 128, 10])

# 2 weights = 2 dim
# but 1, 1, 1 network with biases is 4 dim
# but 1, 1 network with biases is 4 dim
# n.null_biases()


# input, output = train[0], train_labels[0]
# input = np.reshape(input, (784))
# print(input, output)
# print(len(input))
# # n.plot_2d_gradient(input, output, points=50, value=1)
# n.plot_3d_gradient(
#     input,
#     output,
#     path=False
# )
# train = np.reshape(train, (60000, 784))

# train_labels = np.array(train_labels)

# result = np.zeros((train_labels.size, 10), dtype=int)
# result[np.arange(train_labels.size), train_labels] = 1

# train_labels = result

# print(train_labels)

# print(train, print(len(train[0])))

# print(n.get_loss(train, train_labels))
# for _ in range(200):
#     n.learn(train, train_labels, plot=True)

# print(0.5, n.weights[0][0][0], n.activations[0], n.weights[1][0][0], n.activations[1])
n = Network([1, 1, 1])
input = [0.5]
output = [0.5]
print(n.get_variable(n.variable1))
print(n.get_variable(n.variable2))
print(n.get_loss(input, output))
n.plot_2d_gradient(input, output)
n.plot_3d_gradient(input, output, value=0.5)
n.plot_3d_gradient(input, output, value=2)
n.plot_3d_gradient(input, output, value=10)
