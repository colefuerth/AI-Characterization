# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from time import time
from tqdm import tqdm
from functools import cache
from platform import system as ps
from src.zsoc import OCV_curve
from src.dataset import build_dataset, import_k_para

# verify that we are on the GPU
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(
    tf.config.experimental.list_physical_devices('CPU')))


# %% build/load dataset

if not os.path.exists('./res/K_para_ordered.csv'):
    print("K_para_ordered.csv not found. Generating...")
    t = time()
    kp = pd.read_csv('./res/K_para.csv')
    # calculate zsoc for each entry's K0-K7 columns
    kpk = kp.iloc[:, 4:12].to_numpy()
    z = np.array([OCV_curve(k)['Vo'] for k in kpk])
    z = np.sum(z, axis=1)
    # sort kp by zsoc
    kp = kp.iloc[np.argsort(z)]
    kp.to_csv('./res/K_para_ordered.csv')

t, _ = time(), print("Building dataset...")
df = build_dataset(f='./res/K_para_ordered', cache=f'./res/dataset_{ps()}.pkl')
print(f"Done. Took {time()-t:.2f}s.")
kp = import_k_para(f'./res/K_para_ordered.csv')
nsamples = kp.shape[0]
print(f"Dataset has {nsamples} samples.")

# export the first 1000 samples to csv
df[:1000].to_csv('./res/dataset_1000.csv')

t, _ = time(), print("Generating zsoc vectors...")
zsoc = np.array([OCV_curve(kp[i])['Vo'] for i in range(nsamples)])
print(f"Done. Took {time()-t:.2f}s.")

# %% define the model


# class mse_custom(tf.keras.losses.MeanSquaredError):
#     def __init__(self, name='mse_custom'):
#         super().__init__(name=name)

#     def call(self, y_true, y_pred):
#         """mean squared error between zsoc vectors"""
#         y_true = tf.argmax(y_true, axis=-1)
#         y_pred = tf.argmax(y_pred, axis=-1)
#         z_true = tf.gather(zsoc, y_true)
#         z_pred = tf.gather(zsoc, y_pred)
#         return super().call(z_true, z_pred)

# %% generate 3d histograms (preprocessing )

# do a simple 3d histogram of the data using numpy
def hist3dgenerator():
    for i, row in enumerate(df.itertuples()):
        V, I, t = row.V, row.I, row.t
        yield np.histogramdd(
            (V, I, t),
            bins=(20, 20, 20),
            range=((V.min(), V.max()), ((I.min()), (I.max())), (0, t[-1]))
        )[0].reshape(20, 20, 20, 1)


# use tqdm
print("Generating 3d histograms...")
t = time()
cache = f'./res/hist_{ps()}.npy'
if os.path.exists(cache):
    hist = np.load(cache)
else:
    hist = np.array([h for h in tqdm(hist3dgenerator(), total=len(df))])
    np.save(cache, hist)
print(f"Done. Took {time()-t:.2f}s.")


def stupid_model(df: pd.DataFrame):
    """do a dummy tf.keras model that just takes the raw v, i, t vectors and predicts the i"""

    t, _ = time(), print("Building model...")
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(20, 20, 20, 1)))
    model.add(keras.layers.Conv3D(32, (2, 2, 2), activation='relu'))
    model.add(keras.layers.MaxPool3D((3, 3, 3)))
    model.add(keras.layers.Dense(len(zsoc[0]) * 10, activation='sigmoid'))
    model.add(keras.layers.Conv3D(4, (2, 2, 2), activation='relu'))
    model.add(keras.layers.MaxPool3D((3, 3, 3)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(len(zsoc[0]), activation='softmax'))
    model.summary()

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    print(f"Done. Took {time()-t:.2f}s.")

    y = np.array([zsoc[row.i - 1] for row in df.itertuples()])

    model.fit(hist, y, epochs=10, batch_size=16)

    return model


# %% train the model

model = stupid_model(df)
# save the model
model.save('./res/model.h5')
