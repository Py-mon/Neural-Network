from network import Network
from graph import graph
from visualize import NetworkVisualization
from app import CanvasApp
import matplotlib.pyplot as plt
import customtkinter as tk

inputs = (0.5, 0.5)
network = Network((2, 3, 2))

app = CanvasApp(1000, 700, network)

update_line, inputs_, labels = graph(network, lambda x, y: y > abs(x - 0.5))

NetworkVisualization(app, inputs, update_line, inputs_=inputs_, labels=labels)


plt.axis("off")
plt.ion()
plt.show()

app.mainloop()
