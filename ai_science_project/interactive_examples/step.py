from network import Network
from graph import graph
from visualize import NetworkVisualization
from app import CanvasApp
import matplotlib.pyplot as plt


inputs = (0.5, 0.5)
network = Network((2, 3, 2))
network.activation_function = lambda x: (x > 0).astype(int)

app = CanvasApp(1000, 700, network)

update_line, inputs_, labels = graph(network, lambda x, y: y < abs(x - 0.5))
v = NetworkVisualization(
    app,
    inputs,
    update_line,
    inputs_,
    labels,
    learn=False,
)

plt.axis("off")


def main():
    plt.ion()
    plt.show()

    app.mainloop()


if __name__ == "__main__":
    main()
