from network import Network
from graph import graph
from visualize import NetworkVisualization, Neuron, Connection
from app import CanvasApp
import matplotlib.pyplot as plt
import customtkinter as tk
from functions import sigmoid
import time

inputs = (0.5, 0.5)
network = Network((2, 3, 2))
network.activation_function = lambda x: sigmoid(x)

app = CanvasApp(1200, 700, network)

update_line, inputs_, labels = graph(network, lambda x, y: y < x - 0.2)
v = NetworkVisualization(app, inputs, update_line)

Neuron.Max = 25
Neuron.Min = -25
Connection.Max = 25
Connection.Min = -25

def func():
    # print(labels.astype(int))
    # print(network.calculate_outputs(inputs_))
    #print(network.get_loss(inputs_, labels))
    start = time.time()
    for _ in range(10):
        print(network.learn(inputs_, labels))
        
    print(time.time() - start)
    update_line()
    #v.update_network()
    # print(network.get_loss(inputs_, labels))


tk.CTkButton(app, command=func).place(x=0, y=0)

plt.ion()
plt.show()

app.mainloop()

Neuron.Max = 1
Neuron.Min = -1
Connection.Max = 1
Connection.Min = -1
