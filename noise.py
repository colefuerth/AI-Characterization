from noise import pnoise1
import numpy as np

a = np.ones(20)
opensimplex.seed(123)
b = np.array([opensimplex.noise2(j, 0) for j in range(len(a))])
print(b)