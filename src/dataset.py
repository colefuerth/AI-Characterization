# %%

from src.BattSim.BattSim import BattSim
import numpy as np
import pandas as pd
import os
from itertools import product


# %%

def import_k_para(f: str) -> list:
    """Import cached K-parameters from a csv file.

    Args:
        f (str, optional): Path to the csv file. Defaults to 'res/K_para.csv'.

    Returns:
        list: The dataset.
    """
    df_i = pd.read_csv(f)
    df_i = df_i.drop(
        columns=['Sample No.', 'Battery Manufacturer', 'Serial Number', 'Cell Number'])

    # combine columns K0,K1,K2,K3,K4,K5,K6,K7,R0 into a single column as a list of ints
    return np.array(df_i[['K0', 'K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'R0']].values.tolist())


def build_dataset(f: str = 'res/K_para.csv', cache: str = None):
    """Generate a dataset from the csv file.
    If a cache is provided, the dataset will be saved to the cache, or load from it, if it exists.

    Returns:
        pd.DataFrame: The dataset.

    The columns in the dataset are as follows:
    - 'i':int : the index of the K-parameter entry in the dataset (the expected output of the NN)
    - 'Kp':np.ndarray[int] : the corresponding K-parameter in the dataset (the expected output's k-parameters, K0-K7 and R0)
    - 'C':float : The capacity of the cell in Ah
    - 'soc':np.ndarray[float] : The state of charge of the cell for each sample
    - 'I':np.ndarray[float] : The current vector for each sample
    - 'V':np.ndarray[float] : The voltage vector for each sample
    - 't':np.ndarray[float] : The time vector for each sample
    - 'R1C1R2C2':list[float] : The R1, C1, R2, C2 parameters for each sample
    """

    if (cache is not None) and os.path.exists(cache):
        return pd.read_pickle(cache)

    # start by importing the k-parameters
    df = pd.DataFrame()
    # the columns are: ['V':np, 'I', 't', 'kp', 'ki', 'SoC', 'C']
    kp = import_k_para(f)
    print(kp.shape)

    # charge/discharge rates
    C_rates_discharge = np.array([1/16, 1/4, 1/2, 1, 2, 4, 16]) * -1
    C_rates_discharge = np.concatenate(
        [C_rates_discharge, np.array([1/16, 1/4, 1/2, 1, 4])])
    # partials of SoC to sample
    partial_state_of_charge = np.array([1, 0.5, 0.25, 0.1])
    # resolution to scale to
    resolution_arr = np.array([20, 40, 80, 200])

    # Some perlin noise will also be added later for current vectors that are generated incrementally, but for now, just use a constant value

    # for each k-parameter instance in the dataset
    for i, kpi in enumerate(kp):
        # do a cross product between C_rates, state of charge slice size, and resolution
        # place slices' centers on a multimodal distribution, with the peaks on 0.2 and 0.75 soc
        capacity = 1  # set the capacity at a constant 1Ah for now
        # a standard for generating values: (for now we will take the minimum of each range)
        # R1: 100 - 200 m Ohm
        # R2:  100 - 200 m Ohm
        # C1: 5 - 500 F
        # C2: 5 - 500 F
        R1 = 100 * 10**-3
        R2 = 100 * 10**-3
        C1 = 5
        C2 = 5
        ModelID = 2  # 3 for R1C1, 4 for R1C1R2C2
        for C_rate, slice_size, resolution in product(C_rates_discharge, partial_state_of_charge, resolution_arr):
            # first we need to determine the SoC range.
            # we will take the slice_size as a percentage of the total SoC range
            # we will take the center of the slice to be a random value between 0.2 and 0.75, where the slice edges cannot go past the bounds of [0,1]

            # define anchors and choose the index of the anchor to use
            anchors = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
            soc = np.random.multinomial(
                1, [0.1, 0.25, 0.1, 0.1, 0.1, 0.25, 0.1]).argmax()
            # place start of slice at the anchorth position in the possible range of socs
            soc = (1 - slice_size) * anchors[soc]
            if C_rate < 0:
                soc += slice_size

            # now we can generate the current vector
            npoints = int(resolution * slice_size)
            # time in seconds over the sample
            t = np.linspace(0, 3600 * slice_size, npoints)
            # current in A
            I = np.ones(npoints) * C_rate * capacity
            # voltage in V
            V, I, soc, ocv = BattSim(kpi[:-1], capacity, kpi[-1], R1, C1, R2, C2, ModelID, soc=soc).simulate(
                I, t)

            # add the data to the dataframe
            df = df.append({
                'i': i,
                'Kp': kpi,
                'C': capacity,
                'soc': soc,
                'I': I,
                'V': V,
                't': t,
                'R1C1R2C2': [R1, C1, R2, C2]
            }, ignore_index=True)

    if cache is not None:
        df.to_pickle(cache)
    return df

# %%


if __name__ == "__main__":
    # error and return 1
    from sys import stderr
    print("This is a module, not a script. Please import it.", file=stderr)
    exit(1)
