# %%

import numpy as np
import pandas as pd
import os
from itertools import product

# %%
# take a dataset and encode it into a histogram, one histogram per sample and overlay all of them
def encode(df: pd.DataFrame, bins: int = 20, hist_range: tuple = (0, 1), normalize: bool = False, slice_size:float=0.1) -> np.ndarray:
    """
    Encode a dataset into a 3d histogram.
    This histogram is of three dimensions, (V, I, t), the scale of which depend on the sample.

    Args:
        df (pd.DataFrame): The dataset to encode.
        bins (int, optional): The number of bins in the histogram. Defaults to 20. Histogram is size (bins*bins*bins*slices)
        hist_range (tuple, optional): The range of the histogram. Defaults to (0, 1). Only affects the output scaling, to prevent overweighting of high values.
        normalize (bool, optional): Whether to normalize the histograms before merging, as a 3d image. Defaults to False.

    Returns:
        np.ndarray: The encoded dataset.
    """
    # create the bins
    bins = np.linspace(hist_range[0], hist_range[1], bins)
    # create the histogram
    hist = np.zeros((len(df), len(bins)-1))
    # for each sample
    for i, sample in df.iterrows():
        # for each voltage
        for v in sample['V']:
            # add the voltage to the histogram
            hist[i] += np.histogram(v, bins=bins)[0]
    return hist