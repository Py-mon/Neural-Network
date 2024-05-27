from network import Network
from graph import graph
from visualize import NetworkVisualization, Neuron, Connection
from app import CanvasApp
import matplotlib.pyplot as plt
import customtkinter as tk
from functions import sigmoid
import time

Neuron.Max = 25
Neuron.Min = -25
Connection.Max = 25
Connection.Min = -25

inputs = (0.25, 0.25)
network = Network((2, 3, 2))
network.activation_function = sigmoid

app = CanvasApp(1200, 700, network)

update_line, inputs_, labels = graph(network, lambda x, y: y < abs(x - 0.5))
NetworkVisualization(
    app,
    inputs,
    update_line,
    inputs_=inputs_,
    labels=labels,
    learn_rate=1,
    learn_amount=100,
)

plt.axis("off")


def main():
    plt.ion()
    plt.show()

    app.mainloop()

main()

