import numpy as np


def scaling_fwd(x: iter, x_min: float, x_max: float, E: float):
    if type(x) is not np.ndarray:
        x = np.array(x)
    return (1 - 2 * E) * (x - x_min) / (x_max - x_min) + E


def scaling_rev(x: iter, x_min: float, x_max: float, E: float):
    if type(x) is not np.ndarray:
        x = np.array(x)
    return (x - E) * (x_max - x_min) / (1 - 2 * E) + x_min