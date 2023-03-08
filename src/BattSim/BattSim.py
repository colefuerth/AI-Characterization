import subprocess as sp
from math import exp, log
import numpy as np

# although you probably think itll be faster to switch to lists and append onto them, keep in mind numpy vectors are a lot faster and more memory efficient, for when hours of testing are going to happen at a time


class BattSim:
    def __init__(self, Kbatt: list, Cbatt: float, R0: float, R1: float, C1: float, R2: float, C2: float, ModelID: int, soc: float = 0.5):

        # start by checking that available parameters are valid
        if soc < 0 or soc > 1:
            raise ValueError('SoC must be between 0 and 1')
        if ModelID not in [1, 2, 3, 4]:
            raise ValueError('ModelID must be an integer between 1 and 4')
        for k, v in {'Cbatt': Cbatt, 'R0': R0, 'R1': R1, 'C1': C1, 'R2': R2, 'C2': C2}.items():
            v = float(v)
            if (type(v) != float and type(v) != int) or v <= 0:
                raise ValueError(
                    f'{k} must be a positive non-zero number, it was {v} of type {type(v)}!')
        if len(Kbatt) != 8:
            raise ValueError('Kbatt must be a list of length 8')
        # check that the type of x is array-like
        if any(type(x) not in [float, int, np.float32, np.float64] for x in Kbatt):
            raise ValueError('Kbatt must be a list of all numbers')

        # set the parameters
        self.Kbatt = Kbatt
        self.Cbatt = Cbatt
        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.R2 = R2
        self.C2 = C2
        self.ModelID = ModelID

        self.soc = soc

    def __scaling_fwd(self, x, x_min, x_max, E):
        return (1 - 2 * E) * (x - x_min) / (x_max - x_min) + E

    # def __scaling_rev(self, x, x_min, x_max, E):
    #     return (z - E) * (x_max - x_min) / (1 - 2 * E) + x_min

    def simulate(self, I, T, sigma_i=0, sigma_v=10**(-50/20)) -> list:
        """
        Simulate the battery and return the voltage and current

        I: list of current magnitudes in amps
        T: list of timestamps in seconds
        sigma_i: standard deviation of the current in amps
        sigma_v: standard deviation of the voltage in volts

        returns:
        [Vbatt, Ibatt, soc, Vo] as lists of floats
        Vbatt: final output voltage of battery (with sag)
        Ibatt: final output current of battery, same as I
        soc: final state of charge of battery
        Vo: OCV voltage vector
        """
        delta = T[1] - T[0]

        alpha1 = exp(- (delta / (self.R1 * self.C1)))
        alpha2 = exp(- (delta / (self.R2 * self.C2)))

        h = 0

        # initialize the soc list to contain the SoC at any point in time
        # uses timestamped current draws, along with Cbatt (battery capacity in mAh), to determine the SoC.
        # Starts from previous SoC state (this is different from Bala, who starts at 50% always.)
        soc = np.zeros(len(I))
        l = len(soc)
        soc[0] = self.soc
        for k in range(1, l):
            soc[k] = soc[k-1] + 1 / (3600 * self.Cbatt) * \
                I[k] * (T[k] - T[k-1])
            if soc[k] < 0:
                print('Battery is empty!')
                soc[k] = 0
            elif soc[k] > 1:
                print('Battery is full!')
                soc[k] = 1

        # determination of OCV (generate Vo
        Vo = np.zeros(l)  # create Vo (OCV voltage vector)
        # squeeze the beginning and end 0.175 of the SOC curve
        zsoc = self.__scaling_fwd(soc, 0, 1, 0.175)

        for k, zk in enumerate(zsoc):
            Vo[k] = self.Kbatt[0]\
                + self.Kbatt[1] / zk\
                + self.Kbatt[2] / zk ** 2\
                + self.Kbatt[3] / zk ** 3\
                + self.Kbatt[4] / zk ** 4\
                + self.Kbatt[5] * zk\
                + self.Kbatt[6] * log(zk)\
                + self.Kbatt[7] * log(1 - zk)

        # Determine current through R1 and R2

        x1 = np.zeros(l)
        x2 = np.zeros(l)
        for k in range(l-1):
            x1[k+1] = alpha1 * x1[k] + (1 - alpha1) * I[k]
            x2[k+1] = alpha2 * x2[k] + (1 - alpha2) * I[k]

        I1 = np.zeros(l)
        I2 = np.zeros(l)
        for k in range(l-1):
            I1[k] = x1[k+1]

        V = np.zeros(l)

        if self.ModelID == 1:
            V = I * self.R0
        elif self.ModelID == 2:
            V = I * self.R0 + Vo + h
        elif self.ModelID == 3:
            V = I * self.R0 + self.I1 * self.R1 + Vo + h
        elif self.ModelID == 4:
            V = I * self.R0 + self.I1 * self.R1 + self.I2 * self.R2 + Vo + h
        else:
            print('Invalid Model ID')
            return

        V = V + sigma_v * np.random.randn(l)
        I = I + sigma_i * np.random.randn(l)

        return (V, I, soc, Vo)


if __name__ == '__main__':

    Kbatt = [-9.08, 103.087, -18.185, 2.062, -0.102, -76.604, 141.199, -1.117]
    Cbatt = 1.9
    R0 = 0.2
    R1 = 0.1
    C1 = 5
    R2 = 0.3
    C2 = 500
    ModelID = 3

    battSim = BattSim(Kbatt, Cbatt, R0, R1, C1, R2, C2, ModelID)

    from CurrentSIM import *
    I, T = staircase()

    Vbatt, Ibatt, soc, Vo = battSim.simulate(I, T)

    # print(f'Vbatt: {Vbatt}')
    # print(f'Ibatt: {Ibatt}')
    # print(f'SoC: {soc}')
    # print(f'Vo: {Vo}')

    #  plot current/time and voltage/time, stacked
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.grid(visible=True, which='both')
    fig.suptitle('Battery Simulation')
    axs[0].plot(T, Ibatt, label='Ibatt')
    axs[1].plot(T, Vbatt, label='Vbatt')
    axs[0].legend()
    axs[1].legend()
    plt.show()
