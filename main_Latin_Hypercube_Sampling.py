# external imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

rs = 42
np.random.seed(rs)
rng = np.random.default_rng(rs)

# settings
NSIMULATIONS = 10000
sampler = qmc.LatinHypercube(
    d=6,
    rng=rng
)
parameters_array = np.array(sampler.random(n=NSIMULATIONS))

# save parameters array
np.save("data/Latin_Hypercube_Sampling_10000_6.npy", parameters_array)

# plot first two dimensions of parameters
plt.figure(figsize=(8, 8))
plt.scatter(parameters_array[:, 0], parameters_array[:, 1], s=10, alpha=0.5)
plt.xlabel("Parameter 1")
plt.ylabel("Parameter 2")
plt.grid()
plt.show()

# plot only points for which the other dimensions are small
mask = np.all(parameters_array[:, 2:] < 0.1, axis=1)
plt.figure(figsize=(8, 8))
plt.scatter(parameters_array[mask, 0], parameters_array[mask, 1], s=10, alpha=0.5)
plt.xlabel("Parameter 1")
plt.ylabel("Parameter 2")
plt.grid()
plt.show()

