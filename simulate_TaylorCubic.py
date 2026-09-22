import os
import numpy as np
from fenicsx.run_simulation_adaptive import run_simulation_adaptive

from fenicsx.constitutive_models import get_Wbar

np.random.seed(42)


# model info
experiment_name = "plate_with_corner_hole_3D_u2"
disp_max = -2.0
model_name = "TaylorCubic"
strain_energy_density = "theta_1 * (I1bar - 3) + theta_2 * (I2bar32 - 3**(3/2)) + theta_3 * (I1bar - 3)**2 + theta_4 * (I2bar32 - 3**(3/2))**2 + theta_5 * (I1bar - 3)**3 + theta_6 * (I2bar32 - 3**(3/2))**3"

NSIMULATIONS = 3000
NSKIP = 2000
parameters_array = np.load("data/Latin_Hypercube_Sampling_10000_6.npy")

# make directory if it does not exist
if not os.path.exists("fenicsx/data/" + experiment_name):
    os.makedirs("fenicsx/data/" + experiment_name)

# run simulations
for i in range(NSIMULATIONS):

    if i < NSKIP - 1: continue

    parameters = parameters_array[i]
    Wbar = get_Wbar(model_name, parameters)

    print(f"Start simulation {i}.")
    converged, coordinates, connectivity, displacement_data, force_data = run_simulation_adaptive(Wbar,disp_max=disp_max)
    if converged: print("\033[32m", f"Simulation {i} converged.", "\033[0m", sep="")
    else: print("\033[31m", f"Simulation {i} failed.", "\033[0m", sep="")

    # save data
    np.savez(
        "fenicsx/data/" + experiment_name + "/" + model_name + f"_{i:04d}.npz",
        model_name=model_name,
        strain_energy_density=strain_energy_density,
        parameters=parameters,
        converged=converged,
        coordinates=coordinates,
        connectivity=connectivity,
        displacement_data=displacement_data,
        force_data=force_data,
    )
