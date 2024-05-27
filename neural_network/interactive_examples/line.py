from network import Network
from graph import graph
from visualize import NetworkVisualization
from app import CanvasApp
import matplotlib.pyplot as plt
import customtkinter as tk

inputs = (0.25, 0.25)
network = Network((2, 2))

app = CanvasApp(1000, 900, network, 24, 35)

update_line, inputs_, labels = graph(network, lambda x, y: y > x - 0.2) # make key/legend
network_visualization = NetworkVisualization(
    app,
    inputs,
    update_line,
    between_y=500,
    between_x=500,
    neuron_radius=70,
    inputs_=inputs_,
    labels=labels
)

def main():
    plt.ion()
    plt.show()

    app.mainloop()

main()