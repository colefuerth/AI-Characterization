# Current Simulation Utilities
import numpy as np
from itertools import product

# TODO: maybe add another deepdischarge that actually discharges the battery; use perlin noise to generate a discharge pattern

# TODO: each of these has an `Nb` parameter, we should instead generate one 'block' and then repeat it Nb times. This will change these wave generation functions from O(n) to O(1)

# TODO: add an offset parameter, default to 0, for stacking pulses in a stream

def staircase(delta=5, Nsp=5, Ns=5, Nb=4, Imag=[-40, -80, -120, -160], offset=0):
    """
    delta = milliseconds between each step
    Nsp = number of staircase pulses/groups (repeat modifier)
    Ns = samples per block in staircase
    Nb = number of blocks per stairway
    Imag = list of current magnitudes for each step in mA, same number of elements as Nb

    returns:
    (current, time) as lists of floats in amps and seconds
    """
    offset = offset * 10 ** (3) # convert offset to milliseconds
    T = np.arange(offset, offset + Nsp * Ns * delta * Nb, delta)
    T = T * 10 ** (-3) # convert milliseconds to seconds
    I = np.zeros(Nsp * Ns * Nb)

    for k in range(Nb * Nsp):
        I[k * Ns: (k+1) * Ns] = Imag[k % Nb] * np.ones(Ns)

    # I = I * 10 ** (-3) # convert to mA
    return I, T

# TODO: this function is meant to be used with a custom wave; it should be adapted better

def deepdischarge(delta=5, Ns=5, Nb=2, Imag=[0, -1000], offset=0):
    """
    delta = milliseconds time of each step
    Ns = number of steps per block
    Nb = number of blocks
    Imag = list of current magnitudes for each step in mA, same number of elements as Nb
    offset = time offset in seconds

    returns:
    (current, time) as lists of floats in amps and seconds
    """
    offset = offset * 10 ** (3) # convert offset to milliseconds
    T = np.arange(offset, offset + delta * Ns * Nb, delta)
    T = T * 10 ** (-3) # convert milliseconds to seconds

    I = np.zeros(Ns * Nb)

    for k in range(Nb):
        I[k * Ns: (k+1) * Ns] = Imag[k] * np.ones(Ns)
    # I = I * 10 ** (-3) # convert to mA
    return I, T

# TODO: for some reason this is identical to the deepdischarge function???

def rectangular(delta=1000, Ns=500, Nb=2, Imag=[-1000, 0], offset=0):
    """
    delta = milliseconds time of each step
    Ns = number of steps per wave
    Nb = number of waves
    Imag = list of current magnitudes for each step in mA, same number of elements as Nb
    offset = time offset in seconds

    returns:
    (current, time) as lists of floats in amps and seconds
    """
    offset = offset * 10 ** (3) # convert offset to milliseconds
    T = np.arange(offset, delta * Ns * Nb + offset, delta)
    T = T * 10 ** (-3) # convert milliseconds to seconds
    I = np.zeros(Ns * Nb)
    for k in range(Nb):
        I[k * Ns: (k+1) * Ns] = Imag[k] * np.ones(Ns)

    # I = I * 10 ** (-3) # convert to mA
    return I, T

# TODO: this is poorly ported from the old code (rectangular()), needs to be updated
# TODO: maybe add a duty cycle?

def rectangularnew(I1=-0.5, I2=0.5, delta=100*10**(-3), Tc=10, D=100, offset=0):
    """
    delta =  Sampling delta time in seconds
    Tc = Pulse-Width in seconds
    D = Total time in seconds
    I1,I2 = current values for wave halves in Amps
    offset = time offset in seconds

    returns:
    (I, T) as lists of floats in Amps and Seconds
    """

    Np = int(D / Tc)  # Number of on-off pulses
    Nsp2 = int(Tc / delta / 2)  # Number of samples in each half of apulse
    Nb = 2  # Number of blocks = on + off pulse
    Nt = int(D / delta)  # Total number of samples
    Imag = [I1, I2]  # Current vector
    T = np.arange(offset, offset + D, delta)  # Time vector
    I = Imag[0] * np.ones(Nt)  # Current vector
    for k in range(Nb * Np):
        I[k*Nsp2: (k+1)*Nsp2] = Imag[k % Nb] * np.ones(Nsp2)

    # if (Np * Ns_pulse != Nt):
    #     print("Warning: Np * Ns_pulse ~= Nt")
    #     Q = Nt - Np * Ns_pulse
    #     I[k * Nsp2:k * Nsp2 + Q] = I[:Q]

    return I, T



# My own function that I made earlier

from math import log10

# generate a square wave +- 0.5V, at 0.01s intervals, for 10s
def squareWave(amplitude, period, duration, sampleRate=10, offset=0):
    """
    Generate a square wave with the given amplitude, period, duration, and sample rate.
    amplitude = upper and lower limit in Amps
    period = period of a full wave in seconds
    duration = duration of the sample in seconds
    sampleRate = number of samples per second in Hz
    offset = offset of the wave in volts

    returns:
    (current, time) as lists of floats
    """
    a = []
    t = [round(i, round(log10(sampleRate)))
         for i in np.arange(0, duration, 1/sampleRate)]
    for i in t:
        if i % period < period / 2:
            a.append(amplitude + offset)
        else:
            a.append(-amplitude + offset)
    return a, t


def demo():
    import matplotlib.pyplot as plt

    # T, I = staircase()
    # T, I = deepdischarge()
    # T, I = rectangular()
    I, T = rectangularnew()

    # plot the the current (I) vs time (T)
    plt.plot(T, I)
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.show()

if __name__ == "__main__":
    demo()