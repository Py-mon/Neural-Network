from network import Network
from graph import graph
from visualize import NetworkVisualization
from app import CanvasApp
import matplotlib.pyplot as plt
import customtkinter as tk

inputs = (0.5, 0.5)
network = Network((2, 2))

app = CanvasApp(900, 360, network, 24, 35)

update_line, inputs_, labels = graph(network, lambda x, y: y < x - 0.2)
v = NetworkVisualization(app, inputs, update_line, between_x=500)

for i, label in enumerate(labels):
    if label == 1:
        labels[i] = [1,0]
    else:
        labels[i] = [0,1]
        
        
def func():
    # print(labels.astype(int))
    # print(network.calculate_outputs(inputs_))
    print(network.learn(inputs_, labels))
    v.update_network()
    # print(network.get_loss(inputs_, labels))



tk.CTkButton(app, command=func).place(x=0, y=0)

plt.ion()
plt.show()

app.mainloop()
