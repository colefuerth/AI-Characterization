import numpy as np
from src.BattSim.BattSim import BattSim


def derivative(L: np.ndarray, dt: float = 1) -> np.ndarray:
    """
    find the rate of change of a vector
    L: numpy array, volts, dt: float, seconds between datapoints

    returns the rate of change of the vector
    """
    if len(L) < 2:
        return 0
    if type(L) is not np.ndarray:
        L = np.array(L)
    return (L[1:] - L[:-1]) / dt


def integrate_subtract(a: np.ndarray, b: np.ndarray) -> float:
    """
    integrate the absolute value difference of two one-dimensional vectors
    a: numpy array, volts
    b: numpy array, volts

    returns the integrated result
    """
    return np.sum(np.abs(a - b))


def how_straight(a: np.ndarray) -> float:
    """
    metric for how straight a vector of points is

    a: numpy array (2d), integers

    returns int, lower is straighter
    """

    return integrate_subtract(a, np.ones(len(a)) * np.mean(a))


def soc_curve_k(k_param:list, resolution=200) -> tuple[np.ndarray]:
    sim = BattSim(guess_k, 1, *dummy_RC, soc=1.0, ModelID=1)
    delta = 3600 / resolution  # using 200 points on everything
    I = np.ones(resolution) * sim.Cbatt * -1
    T = np.arange(0, 3600, delta)
    Vbatt, Ibatt, soc, Vo = sim.simulate(I, T, sigma_v=0)
    return Vo, soc