import numpy as np

from scipy.special import softmax

from functions import sigmoid, random
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class Network:
    def __init__(self, layers, step=False, sigmoid=False, softmax=False):

        self.neuron_layers = layers
        self.num_layers = len(layers)
        self.inputs = layers[0]
        self.outputs = layers[-1]

        self.randomize_weights_biases(layers)

        self.loss_gradient_weights = [np.empty_like(x) for x in self.weights]
        self.loss_gradient_biases = [np.empty_like(x) for x in self.biases]

        self.sigmoid = sigmoid
        self.softmax = softmax
        self.step = step

        self.variable1: tuple[list, int, int, int] = (self.weights, 0, 0, 0)
        self.variable2: tuple[list, int, int, int] = (self.weights, 1, 0, 0)

        self.path = []

    def randomize_weights_biases(self, layers):
        self.biases = [
            random(y) for y in layers[1:]
        ]  # No biases in the input layer, hence the '[1:]'

        self.weights: list[np.ndarray] = [
            random(y, x) for x, y in zip(layers[:-1], layers[1:])
        ]

    def activation_function(self, output):
        return output

    def activation_dir(self, output):
        return output

    def loss_dir(self, prediction, label):
        return 2 * (prediction - label)

    def calculate_output(self, input):
        self.activations = []
        self.weighted_inputs = []

        def compute_layer(input, layer):
            bias, weight = layer

            weighted_inputs = np.dot(weight, input) + bias
            activation = self.activation_function(weighted_inputs)

            self.weighted_inputs.append(weighted_inputs)
            self.activations.append(activation)

            return activation

        result = reduce(compute_layer, zip(self.biases, self.weights), input)

        return result

    def calculate_outputs2(self, inputs):
        self.activations = []
        self.weighted_inputs = []

        def cal(input):
            activations = []
            weighted_inputs = []

            def compute_layer(input, layer):
                bias, weight = layer

                weighted_inputs1 = np.dot(weight, input) + bias
                activation = self.activation_function(weighted_inputs)

                weighted_inputs.append(weighted_inputs1)
                activations.append(activation)

                return activation

            result = reduce(compute_layer, zip(self.biases, self.weights), input)

            self.weighted_inputs.append(weighted_inputs)
            self.activations.append(activations)

            return result

        return np.apply_along_axis(cal, -1, inputs)

    def calculate_outputs_graph(self, inputs):
        return np.apply_along_axis(
            lambda x: x[0] < x[1], -1, self.calculate_outputs(inputs)
        )

    def calculate_outputs(self, inputs):
        return np.apply_along_axis(self.calculate_output, -1, inputs)

    def loss(self, prediction, label):
        return (prediction - label) ** 2  # positive (emphasize differences)

    def get_loss(self, inputs, labels):
        outputs = self.calculate_outputs(inputs)

        return np.average(self.loss(outputs, labels))

    def cal_learn(self, input, labels, learn_rate=0.1):
        outputs = self.calculate_outputs2(input)
        node_values = self.loss_dir(self.activations, labels) * self.activation_dir(
            self.weighted_inputs
        )
        gradientB = node_values
        gradientW = np.dot(outputs, node_values)

        for i in range(self.num_layers - 1):
            self.biases[i] -= gradientB[i] * learn_rate
            self.weights[i] -= gradientW[i] * learn_rate

        loss = self.get_loss(input, labels)
        return loss

    def learn(self, input, labels, learn_rate=0.1):
        increment = 0.000001
        original_loss = self.get_loss(input, labels)

        for layer_i in range(self.num_layers - 1):
            layer_weights = self.weights[layer_i]
            for i in range(len(layer_weights)):
                weights = layer_weights[i]
                for j in range(len(weights)):
                    weights[j] += increment
                    loss = self.get_loss(input, labels)
                    change_in_loss = loss - original_loss
                    weights[j] -= increment

                    self.loss_gradient_weights[layer_i][i][j] = (
                        change_in_loss / increment
                    )

            biases = self.biases[layer_i]
            for i in range(len(biases)):
                biases[i] += increment
                change_in_loss = self.get_loss(input, labels) - original_loss
                biases[i] -= increment
                self.loss_gradient_biases[layer_i][i] = change_in_loss / increment

        for i in range(self.num_layers - 1):
            self.biases[i] -= self.loss_gradient_biases[i] * learn_rate
            self.weights[i] -= self.loss_gradient_weights[i] * learn_rate

        loss = self.get_loss(input, labels)
        return loss

    def calculate_losss_of_weights(
        self,
        input,
        output,
        points,
        range_,
    ):
        variable1s = np.linspace(-range_, range_, points)
        variable2s = np.linspace(-range_, range_, points)

        # x, y, z = (
        #     np.empty(points**2),
        #     np.empty(points**2),
        #     np.empty((points, points)),
        # )
        x, y, z = (
            np.empty((points, points)),
            np.empty((points, points)),
            np.empty((points, points)),
        )
        for i, variable1 in enumerate(variable1s):
            for j, variable2 in enumerate(variable2s):
                parameter, layer_index, *index = self.variable1
                parameter[layer_index][index] = variable1
                parameter, layer_index, *index = self.variable2
                parameter[layer_index][index] = variable2
                loss = self.get_loss(input, output)
                # index = i * points + j
                x[i, j] = variable1
                y[i, j] = variable2
                z[i, j] = loss
        return x, y, z

    def plot_2d_gradient(self, input, labels, points=500, value=10):
        x, y, z = self.calculate_losss_of_weights(
            input,
            labels,
            points,
            value,
        )
        # Plot Image

        plt.imshow(np.log10(z), cmap="coolwarm", interpolation="bilinear")

        ax = plt.gca()
        ax.set_yticklabels(np.round(np.linspace(min(y), max(y), 10)))
        ax.set_xticklabels(np.round(np.linspace(min(x), max(x), 10)))

        ax.set_xlabel("weight1")
        ax.set_ylabel("weight2")

        plt.show()

    def plot_3d_gradient(
        self,
        input,
        labels,
        pathX,
        pathY,
        pathZ,
        points=500,
        value=10,
    ):
        x, y, z = self.calculate_losss_of_weights(
            input,
            labels,
            points,
            value,
        )
        # Plot Image

        # nx, ny = points, points
        # x = range(nx)
        # y = range(ny)

        hf = plt.figure()
        ha = hf.add_subplot(111, projection="3d")

        # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D
        ha.plot_surface(x, y, z, cmap="coolwarm")
        # import matplotlib.ticker as ticker
        # ha.yaxis.set_major_locator(ticker.IndexLocator(1, 0))
        print(value)
        # ha.xaxis.set_ticks(np.linspace(0, points, 6))
        # ha.yaxis.set_ticks(np.linspace(0, points, 6))
        ha.set_ylabel("Weight 2")
        ha.set_xlabel("Weight 1")
        ha.set_zlabel("Loss")
        # ha.set_yticklabels([str(x) for x in np.round(np.linspace(-value, value, 6), 2)])
        # ha.set_xticklabels([str(x) for x in np.round(np.linspace(-value, value, 6), 2)])

        print(pathX, pathY)
        ha.plot(pathX, pathY, pathZ, c="red", alpha=1, linewidth=3)
        # ha.scatter3D(pathX, pathY, pathZ, c='red', alpha=1)

        plt.show()

    def null_biases(self):
        """nullifies the biases to 0"""
        self.biases = np.zeros_like(self.biases)


# n = Network([2, 3, 2])
# n.activation_function = sigmoid
# inputs = np.array([[0.5, 0.5], [1, 1]])
# output = np.array([[1, 0], [0, 1]])
# n.plot_2d_gradient(inputs, output, 20)
# n.plot_3d_gradient(inputs, output, 50)

# # Gradient Descent Example
# network = Network([1, 1, 1])  #  o-o-o network
# input = np.array([0.5])
# expected_result = np.array([1])
# network.null_biases()

# network.weights[0][0][0] = -2
# network.weights[1][0][0] = 1.9

# pathX = []
# pathY = []
# pathZ = []
# for i in range(100):
#     network.learn(input, expected_result, 0.1)
#     network.null_biases()
#     pathX.append(network.weights[0][0][0])
#     pathY.append(network.weights[1][0][0])
#     pathZ.append(network.get_loss(input, expected_result))

# print(pathX, pathY)

# print(network.get_loss(input, expected_result))
# network.plot_3d_gradient(input, expected_result, pathX, pathY, pathZ, value=2)


# input = np.array([0.5])
# output = np.array([0.5])
# # x = n.calculate_output(input)
# # print(x)
# # print(n.get_loss(x, output))
# for i in range(1):
#     print(n.learn(input, output))
# print(n.get_loss(x, output))
# n.null_biases()
# n.plot_2d_gradient(input, output)
