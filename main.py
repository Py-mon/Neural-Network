from network import Network
from line import line
from visualize import NetworkVisualization
from app import CanvasApp
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import customtkinter as tk
from matplotlib.animation import FuncAnimation


inputs = (0.5, 0.5)
network = Network((2, 1))
app = CanvasApp(1000, 700, network)

update_line, acc, input, output, fig = line(network)
v = NetworkVisualization(app, inputs, update_line)


# network.plot_2d_gradient(input, output)


def func3(*args):
    network.learn(input, output, 0.3)
    print(network.activations[-1])
    # x += 0.1
    update_line()
    print(network.get_avg_cost(input, output))


# amin = FuncAnimation(fig, func3, interval=1)

plt.ion()
plt.show()


# https://stackoverflow.com/questions/10077644/how-to-display-text-with-font-and-color-using-pygame


# for i in range(500):
#     network.learn(input, output, 1)
#     print(network.weights, network.biases)
#     plt.pause(0.01)

#     update_line()
# v.update_network2()


def func2():
    print(network.get_avg_cost(input, output))

    update_line()


tk.CTkButton(app, command=func2).place(x=0, y=100)

print(input, output)


def func():
    print("learning")
    network.learn(input, output, 1)
    print(network.get_avg_cost(input, output))

    update_line()
    v.update_network2()


tk.CTkButton(app, command=func).place(x=0, y=0)

# print(network.calculate_output((0, 0)))
# print(network.calculate_output((0.50, 0.50)))
# print(network.calculate_output((0.20, 1)))
# print(network.calculate_output((1, 0.20)))
# plt.scatter(*np.array(((0, 0), (0.50, 0.50), (0.20, 0.9), (1, 0.20))).T, s=200)
# print(acc())

app.mainloop()

# root = tk.CTk()

# WIDTH, HEIGHT = 1000, 700

# root.geometry(f"{WIDTH}x{HEIGHT}")

# background = root.cget("background")
# canvas = tk.CTkCanvas(
#     width=WIDTH,
#     height=HEIGHT,
#     bg=background,
#     bd=0,
#     highlightthickness=0,
#     relief="ridge",
# )
# text = CanvasObject(canvas.create_text(100, 500))

# font = tk.CTkFont(family="font1", size=24, weight="bold")
# font2 = tk.CTkFont(family="font1", size=12, weight="bold")

# inputs = [0.5, 0.5]
# network = Network([2, 3, 1])
# print(network.biases, network.weights)

# # dieses - study patents - food, cough,sneezing - heart rate, steps
# # New diese you want to prevent


# set_canvas(canvas, font)

# neurons, connections, biases = visualize_network(network, inputs, font, font2, font2)
# f2, acc = line(network, inputs)


# def f3():
#     f2()
#     text.edit(text=str(acc()))


# slide_network(connections, biases, canvas, network, inputs, neurons, font2, f3)

# canvas.pack()

# root.mainloop()
