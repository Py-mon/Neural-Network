from network import Network
from graph import graph
from visualize import NetworkVisualization
from app import CanvasApp
import matplotlib.pyplot as plt


inputs = (0.5, 0.5)
network = Network((2, 2))
network.activation_function = lambda x: (x > 0).astype(int)

app = CanvasApp(1000, 700, network)

update_line = graph(network, lambda x, y: y < abs(x - .5))
v = NetworkVisualization(app, inputs, update_line)

plt.axis('off')
plt.ion()
plt.show()

app.mainloop()
