import numpy as np


def grayscale_from_value(value, min_, max_):
    """Return a rgb of grayscale from a value between a min and max"""

    def normalize(value):
        return (value - min_) / (max_ - min_)

    percent = normalize(value)
    make_in_bounds = min(max(percent, min_), max_)
    shade = max(0, round(make_in_bounds * 200))

    def from_rgb(rgb):
        return "#%02x%02x%02x" % rgb

    return from_rgb((shade, shade, shade))


def round3(value):
    return np.round(value, 3)
