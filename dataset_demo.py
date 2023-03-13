# %%
from src.dataset import build_dataset, import_k_para
from platform import system as ps
from time import time
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from functools import cache
from src.zsoc import OCV_curve
# import cupy as cp

# %%

t = time()
print("Building dataset...")
df = build_dataset(cache=f'./res/dataset_{ps()}.pkl')
print(f"Done. Took {time()-t:.2f}s.")
# df = build_dataset()
# df.to_csv('./res/dataset.csv')
kp = import_k_para(f'./res/K_para.csv')
nsamples = kp.shape[0]
print(f"Dataset has {nsamples} samples.")

print("Generating zsoc vectors...")
t = time()
z = np.array([OCV_curve(kp[i])['Vo'] for i in range(nsamples)])
print(f"Done. Took {time()-t:.2f}s.")

# %%

def mse(y_true, y_pred):
    """mean squared error for keras using the zsoc vectors"""
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    z_true = tf.gather(z, y_true)
    z_pred = tf.gather(z, y_pred)
    return tf.reduce_mean(tf.square(z_true - z_pred))


def stupid_model(df: pd.DataFrame):
    """do a dummy tf.keras model that just takes the raw v, i, t vectors and predicts the i"""
    
    # do a simple 3d histogram of the data using numpy
    def hist3dgenerator():
        for i, row in enumerate(df.itertuples()):
            yield np.histogramdd(
                (row.V, row.I, row.t),
                bins=(20, 20, 20),
                range=((2.5, 4.2), ((row.I.min), (row.I.max)), (0, row.t[-1])),
                density=True
            )[0].reshape(20, 20, 20, 1)

    # use tqdm
    print("Generating 3d histograms...")
    t = time()
    cache = f'./res/hist_{ps()}.npy'
    if os.path.exists(cache):
        hist = np.load(cache)
    else:
        hist = np.array([h for h in tqdm(hist3dgenerator())])
        np.save(cache, hist)
    print(f"Done. Took {time()-t:.2f}s.")

    print("Building model...")
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(20, 20, 20, 1)))
    model.add(keras.layers.Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(keras.layers.MaxPool3D((2, 2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(nsamples, activation='linear'))
    model.summary()

    model.compile(
        optimizer='adam',
        loss=mse,
        metrics=[mse]
    )

    print("Done.")
    # one-hot encode the labels (df.i)
    i = df.i
    i = np.eye(nsamples)[i - 1]
    model.fit(hist, i, epochs=10, batch_size=32)

    return model

model = stupid_model(df)

