import numpy as np

from scipy.special import softmax

from functions import sigmoid, random
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib import cm


class Network:
    def __init__(self, layers, step=False, sigmoid=False, softmax=False):

        self.neuron_layers = layers
        self.num_layers = len(layers)
        self.inputs = layers[0]
        self.outputs = layers[-1]

        self.randomize_weights_biases(layers)

        self.cost_gradient_weights = [np.empty_like(x) for x in self.weights]
        self.cost_gradient_biases = [np.empty_like(x) for x in self.biases]

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

    def _calculate_output(self, input):
        self.activations = []

        def compute_layer(input, layer):
            bias, weight = layer

            activation = np.dot(weight, input) + bias

            self.activations.append(activation)

            return activation

        return reduce(compute_layer, zip(self.biases, self.weights), input)

    def calculate_output(self, input):
        return self._calculate_output(input)

    def binary_output(self, input):
        output = (self._calculate_output(input) > 0).astype(int)
        self.activations[-1] = (self.activations[-1] > 0).astype(int)
        return output

    def calculate_outputs(self, inputs):
        return np.apply_along_axis(self.calculate_output, -1, inputs)

    def cost(self, prediction, label):
        return (prediction - label) ** 2  # positive (emphasize differences)

    def get_cost(self, input, label):
        output = self.calculate_output(input)

        total = 0
        for out, lab in zip(output, label):
            total += self.cost(out, lab)
        return total

    def get_avg_cost(self, inputs, labels):
        total = 0
        for input, label in zip(inputs, labels):
            total += self.get_cost(input, label)

        return total / len(inputs)

    def learn(self, input, labels, learn_rate=1):
        increment = 0.000001
        # increment = 0.1
        original_cost = self.get_avg_cost(input, labels)

        for layer_i in range(self.num_layers - 1):
            for i, weights in enumerate(self.weights[layer_i]):
                for j, _ in enumerate(weights):
                    self.weights[layer_i][i][j] += increment
                    cost = self.get_avg_cost(input, labels)
                    change_in_cost = cost - original_cost
                    self.weights[layer_i][i][j] -= increment

                    self.cost_gradient_weights[layer_i][i][j] = (
                        change_in_cost / increment
                    )

            for i, _ in enumerate(self.biases[layer_i]):
                self.biases[layer_i][i] += increment
                change_in_cost = self.get_avg_cost(input, labels) - original_cost
                self.biases[layer_i][i] -= increment
                self.cost_gradient_biases[layer_i][i] = change_in_cost / increment

        for i in range(self.num_layers - 1):
            self.biases[i] -= self.cost_gradient_biases[i] * learn_rate
            self.weights[i] -= self.cost_gradient_weights[i] * learn_rate

        cost = self.get_avg_cost(input, labels)
        return cost

    def calculate_costs_of_weights(
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
                cost = self.get_cost(input, output)
                index = i * points + j
                x[index] = variable1
                y[index] = variable2
                z[i, j] = cost
        return x, y, z

    def plot_2d_gradient(self, input, labels, points=500, value=10):
        """Satisfies the xy = 0.5 equation.
        ```
            |
        ___  \\___
        \\
            |
            """
        x, y, z = self.calculate_costs_of_weights(
            input,
            labels,
            points,
            value,
        )
        # Plot Image
        plt.imshow(np.log10(z), cmap=cm.binary, interpolation="bilinear")

        ax = plt.gca()
        ax.set_yticklabels(np.round(np.linspace(min(y), max(y), 10)))
        ax.set_xticklabels(np.round(np.linspace(min(x), max(x), 10)))

        ax.set_xlabel("weight1")
        ax.set_ylabel("weight2")

        plt.show()

    def null_biases(self):
        """nullifies the biases to 0"""
        self.biases = np.zeros_like(self.biases)


# n = Network([1, 1, 1])
# print(n.weights, n.biases)
# input = [0.5]
# output = [0.5]
# print(n.calculate_output(input))
# print(n.get_avg_cost([[0.5], [1], [2]], [[0.5], [1], [2]]))

# n.null_biases()
# n.plot_2d_gradient(input, output)