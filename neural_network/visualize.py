import numpy as np
import matplotlib.pyplot as plt
import customtkinter as tk
from math import atan2, pi
from utils import grayscale_from_value, round3
from app import CanvasApp


class InputNeuron:
    Max = 1
    Min = -1

    def __init__(self, x: int, y: int, radius: int, app: CanvasApp):
        self.x = x
        self.y = y
        self.radius = radius
        self.app = app

        OUTLINE_WIDTH = 4

        self.circle = self.app.canvas.create_oval(
            x + radius,
            y + radius,
            x - radius,
            y - radius,
            width=OUTLINE_WIDTH,
            outline="white",
        )
        self.radius = self.radius + OUTLINE_WIDTH // 2

        label_font = self.app.create_new_font(24)
        self.shadow = self.app.canvas.create_text(
            x - 2, y - 2, font=label_font, fill="black"
        )

        self.label = self.app.canvas.create_text(x, y, font=label_font, fill="white")

        self.bias_label = self.app.canvas.create_text(
            x + self.radius,
            y + self.radius,
            font=self.app.create_new_font(12),
            fill="white",
        )

    def set_activation(self, activation: float):
        rounded = round3(activation)

        self.app.canvas.edit(self.label, text=rounded)
        self.app.canvas.edit(self.shadow, text=rounded)

        self.app.canvas.edit(
            self.circle,
            fill=grayscale_from_value(activation, type(self).Min, type(self).Max),
        )


class Neuron(InputNeuron):
    """A Neuron with a bias"""

    def __init__(
        self,
        x: int,
        y: int,
        radius: int,
        app: CanvasApp,
        bias: tuple[int, int] | None = None,
    ):
        super().__init__(x, y, radius, app)

        if bias is not None:
            self._bias = bias
            i, j = self._bias
            self.set_bias(self.app.network.biases[i][j])

    @property
    def bias(self):
        i, j = self._bias
        return self.app.network.biases[i][j]

    def set_bias(self, bias: float):
        self.app.canvas.edit(self.bias_label, text=round3(bias))

        i, j = self._bias
        self.app.network.biases[i][j] = bias


class Connection:
    Max = 1
    Min = -1

    def __init__(
        self,
        neuron1: Neuron,
        neuron2: Neuron,
        weight: tuple[int, int, int],
        app: CanvasApp,
        above=True,
    ):
        self.neuron1 = neuron1
        self.neuron2 = neuron2
        self._weight = weight

        self.app = app

        self.x_center = (neuron1.x + neuron2.x) // 2
        self.y_center = (neuron1.y + neuron2.y) // 2

        left_edge = neuron1.x + neuron1.radius
        right_edge = neuron2.x - neuron2.radius

        self.line = (
            self.app.canvas.create_line(
                (left_edge, neuron1.y),
                (right_edge, neuron2.y),
                width=4,
                dash=15,
            ),
        )

        x1, y1 = (left_edge, neuron1.y)
        x2, y2 = (right_edge, neuron2.y)
        angle = -180 / pi * atan2(y2 - y1, x2 - x1)

        label_offset = 10
        y = self.y_center + label_offset
        if above:
            y -= label_offset * 2

        self.label = (
            self.app.canvas.create_text(
                self.x_center,
                y,
                angle=angle,
                font=self.app.create_new_font(12),
                fill="white",
            ),
        )

        i, j, k = self._weight
        self.set_weight(self.app.network.weights[i][j][k])

    @property
    def weight(self):
        i, j, k = self._weight
        return self.app.network.weights[i][j][k]

    def set_weight(self, weight):
        self.app.canvas.edit(self.label, text=round3(weight))

        self.app.canvas.edit(
            self.line, fill=grayscale_from_value(weight, type(self).Min, type(self).Max)
        )

        i, j, k = self._weight
        self.app.network.weights[i][j][k] = weight


class NetworkVisualization:
    def __init__(
        self,
        app: CanvasApp,
        input_visual: tuple[float, ...],
        extra_func,
        inputs_=None,
        labels=None,
        between_y=100,
        between_x=300,
        neuron_radius=50,
        padding=100,
        loss=True,
        learn=True,
        learn_rate=0.5,
        learn_amount=1,
    ) -> None:
        self.app = app

        self.network = self.app.network
        self.input_visual = input_visual

        self.between_y = between_y
        self.between_x = between_x

        self.neuron_radius = neuron_radius
        self.padding = padding

        self.extra_func = extra_func

        self.app.network.calculate_output(self.input_visual)

        max_layer_size = max(self.app.network.neuron_layers)
        even_size_layers: bool = min(self.app.network.neuron_layers) == max_layer_size

        def layer_offset(layer):
            height_of_max_layer = padding + between_y * max_layer_size
            layer_size = len(layer)

            return (height_of_max_layer - (2 * layer_size * self.neuron_radius)) / (
                layer_size + 1
            ) + padding

        x = padding
        y = padding
        if not even_size_layers:
            y = layer_offset(self.input_visual)

        input_neurons = []

        for input in self.input_visual:
            neuron = InputNeuron(x, y, self.neuron_radius, self.app)
            neuron.set_activation(input)
            input_neurons.append(neuron)

            y += between_y + neuron.radius

        self.neurons = []
        self.weights = []
        connections = {}
        biases = {}
        for i, layer in enumerate(self.app.network.activations):
            y = padding
            if len(layer) != max_layer_size and not even_size_layers:
                y = layer_offset(layer)

            x += between_x

            next_neurons = []
            layer_connections = []
            weights = []
            for j, activation in enumerate(layer):
                neuron = Neuron(x, y, self.neuron_radius, self.app, (i, j))
                neuron.set_activation(activation)
                next_neurons.append(neuron)

                layer_weights = []

                for k, input in enumerate(input_neurons):
                    might1 = (
                        j * len(layer) + k * len(input_neurons) % len(input_neurons)
                        == 0
                    )
                    might2 = j > k
                    might3 = k > j

                    c = Connection(input, neuron, (i, j, k), self.app, might2)
                    layer_weights.append(c)
                    layer_connections.append(c)

                weights.append(layer_weights)

                y += between_y + neuron.radius

            input_neurons = next_neurons
            self.neurons.append(input_neurons)

            self.weights.append(weights)

            biases[f"Biases Layer {i+1}"] = input_neurons

            layer_name = f"Weights Layer {i+1}"
            connections[layer_name] = layer_connections

        self.app.create_slider_menu(connections, biases, self.update_network)

        def func():
            for _ in range(learn_amount):
                # print('learning!')
                self.app.network.learn(inputs_, labels, learn_rate)

            self.update_network()

        self.doLoss = loss
        self.inputs_ = inputs_
        self.labels = labels

        if self.doLoss:
            self.loss = tk.StringVar()

            tk.CTkLabel(
                app, textvariable=self.loss, font=self.app.create_new_font(24)
            ).place(x=self.app.width // 2, y=self.app.height - 50)

            self.loss.set(
                "Loss: "
                + str(round(self.app.network.get_loss(self.inputs_, self.labels), 5))
            )

        if learn:
            tk.CTkButton(app, command=func, text="Learn").place(
                x=200, y=self.app.height - 50
            )

    def update_network(self):
        self.app.network.calculate_output(self.input_visual)

        if self.doLoss:
            self.loss.set(
                "Loss: "
                + str(round(self.app.network.get_loss(self.inputs_, self.labels), 5))
            )

        for i, layer in enumerate(self.app.network.activations):
            for j, activation in enumerate(layer):
                self.neurons[i][j].set_activation(activation)

        self.extra_func()

        for i, layer in enumerate(self.app.network.biases):
            for j, activation in enumerate(layer):
                self.neurons[i][j].set_bias(activation)

        for i, layer in enumerate(self.app.network.weights):
            for j, activation in enumerate(layer):
                for k, activation in enumerate(activation):
                    self.weights[i][j][k].set_weight(activation)


# def visualize_network(
#     network: Network,
#     inputs,
#     neuron_font,
#     connection_font,
#     bias_label_font,
#     between_y=100,
#     between_x=300,
#     radius=50,
#     padding=100,
# ):
#     Neuron.label_font = neuron_font
#     Neuron.bias_label_font = bias_label_font
#     Neuron.network = network
#     Connection.label_font = connection_font
#     Connection.network = network

#     network.calculate_output(inputs)

#     max_layer_size = max(network.neuron_layers)
#     even_size_layers: bool = min(network.neuron_layers) == max_layer_size

#     def layer_offset(layer):
#         height_of_max_layer = padding + between_y * max_layer_size
#         layer_size = len(layer)

#         return (height_of_max_layer - (2 * layer_size * radius)) / (
#             layer_size + 1
#         ) + padding

#     x = padding
#     y = layer_offset(inputs)

#     input_neurons = []

#     for input in inputs:
#         neuron = InputNeuron(x, y, radius, network)
#         neuron.activation = input
#         input_neurons.append(neuron)

#         y += between_y + neuron.radius

#     neurons = []
#     connections = {}
#     biases = {}
#     for i, layer in enumerate(network.activations):
#         y = padding
#         if len(layer) != max_layer_size and not even_size_layers:
#             y = layer_offset(layer)

#         x += between_x

#         next_neurons = []
#         layer_connections = []
#         for j, activation in enumerate(layer):
#             neuron = Neuron(x, y, radius, (i, j))
#             neuron.activation = activation
#             next_neurons.append(neuron)

#             for k, input in enumerate(input_neurons):
#                 might1 = (
#                     j * len(layer) + k * len(input_neurons) % len(input_neurons) == 0
#                 )
#                 might2 = j > k
#                 might3 = k > j

#                 c = Connection(input, neuron, (i, j, k), might2)
#                 layer_connections.append(c)

#             y += between_y + neuron.radius

#         input_neurons = next_neurons
#         neurons.append(input_neurons)

#         biases[f"Layer Biases {i+1}"] = input_neurons

#         layer_name = f"Layer Weights {i+1}"
#         connections[layer_name] = layer_connections

#     return neurons, connections, biases


# def update_network(network: Network, inputs, neurons):
#     network.calculate_output(inputs)

#     for i, layer in enumerate(network.activations):
#         for j, activation in enumerate(layer):
#             neurons[i][j].activation = activation


# def slide_network(
#     connections, biases, root, network, inputs, neurons, category_font, extra_command
# ):
#     SliderMenu(
#         connections,
#         biases,
#         root,
#         category_font,
#         lambda: (
#             update_network(
#                 network,
#                 inputs,
#                 neurons,
#             ),
#             extra_command(),
#         ),
#     )
