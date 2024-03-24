import customtkinter as tk
from network import Network


class Canvas(tk.CTkCanvas):
    def edit(self, id, **kwargs):
        self.itemconfig(id, **kwargs)

    def get(self, id, item):
        return self.itemcget(id, item)


class CanvasApp(tk.CTk):
    def __init__(self, width, height, network: Network):

        tk.CTk.__init__(self)

        self.font = tk.CTkFont(family="font1", weight="bold")

        self.width, self.height = width, height
        self.geometry(f"{self.width}x{self.height}")

        background = self.cget("background")
        self.canvas = Canvas(
            width=self.width,
            height=self.height,
            bg=background,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas.pack()

        self.network = network

    def create_new_font(self, size):
        font = self.font.actual()
        font["size"] = size
        return tk.CTkFont(**font)

    def create_slider_menu(self, connections, neurons: dict, update_neurons_func):
        padding = 20

        width, height = 200, self.height - (padding * 2)

        frame = tk.CTkFrame(self, corner_radius=30, width=width, height=height)

        def add_category(division):
            category = tk.CTkLabel(frame, font=self.create_new_font(24), text=division)

            frame.update_idletasks()

            category.place(x=width // 2 - (category.winfo_reqwidth() // 2), y=y)

        padding = 20
        y_increment = 35
        y = 35

        def create_weight_func(connection):
            def f(x):
                connection.set_weight(x)
                update_neurons_func()

            return f

        for division, connections in connections.items():
            add_category(division)

            y += y_increment

            for neuron in connections:
                slider = tk.CTkSlider(
                    frame,
                    from_=neuron.MIN,
                    to=neuron.MAX,
                    width=width - (padding * 2),
                    command=create_weight_func(neuron),
                )
                slider.set(neuron.weight)
                slider.place(x=padding, y=y)
                y += y_increment

        def create_bias_func(connection):
            def f(x):
                connection.set_bias(x)
                update_neurons_func()

            return f

        for division, neurons in neurons.items():
            add_category(division)

            y += y_increment

            for neuron in neurons:
                slider = tk.CTkSlider(
                    frame,
                    from_=-1,
                    to=1,
                    width=width - (padding * 2),
                    command=create_bias_func(neuron),
                )
                slider.set(neuron.bias)
                slider.place(x=padding, y=y)
                y += y_increment

        self.canvas.create_window(
            self.width - padding - (width // 2),
            padding + (height // 2),
            window=frame,
        )

