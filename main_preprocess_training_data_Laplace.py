# external imports
import numpy as np
import os
import torch

rs = 42
np.random.seed(rs)
torch.manual_seed(rs)

# custom imports
import neural_operators as no
from neural_operators.fem_utils import assemble_K_M
from fenicsx.constitutive_models import get_Wbar

# settings
experiment_name = "plate_with_corner_hole_3D_u2"
n_eigen = 100
Ibar_sampling_name = "grid"
# Ibar_sampling_name = "grid_linear"
Ibar_range = 5
n_Ibar_sampling = 100
normalize_force = True
training_name = "training_Taylor"
training_name += "_Laplace_" + str(int(n_eigen))
training_name += "_" + Ibar_sampling_name + "_" + str(int(Ibar_range))

# LIST OF SIMULATIONS TO LOAD
model_list = [
    "TaylorCubic",
]
simulation_list = []
if "TaylorCubic" in model_list:
    for i in range(3000):
        simulation_list.append("TaylorCubic" + f"_{i:04d}")

# INVARIANTS SAMPLING
I1bar_train, I2bar32_train = no.data_utils.sample_Ibar(
    Ibar_sampling_name=Ibar_sampling_name,
    normalize = False,
    Ibar_range = Ibar_range,
    n_Ibar_sampling = n_Ibar_sampling,
)
n_invariants = I1bar_train.shape[0]


# LOAD SIMULATION DATA
displacement_nn_data = []
force_nn_data = []
I1bar_nn_data = []
I2bar32_nn_data = []
Wbar_nn_data = []
n_ignored = 0
for simulation in simulation_list:
    try:
        data = np.load(
            "fenicsx/data/" + experiment_name + "/" + simulation + ".npz",
            allow_pickle=False
            )
    except:
        print("The following simulation was not found:", simulation)
        n_ignored += 1
        continue
    if not data["converged"]:
        print("The following simulation did not converge:", simulation)
        n_ignored += 1
        continue
    Wbar = get_Wbar(data["model_name"], data["parameters"])

    displacement_nn_data.append(data["displacement_data"])
    force_nn_data.append(data["force_data"])
    I1bar_nn_data.append(I1bar_train)
    I2bar32_nn_data.append(I2bar32_train)
    Wbar_nn_data.append(Wbar(I1bar_train, I2bar32_train))

    if normalize_force:
        # normalize force and Wbar data
        force_norm = np.linalg.norm(force_nn_data[-1])
        force_nn_data[-1] = force_nn_data[-1] / force_norm
        Wbar_nn_data[-1] = Wbar_nn_data[-1] / force_norm


# CONVERT LISTS TO NUMPY ARRAYS
displacement_nn_data = np.array(displacement_nn_data)
force_nn_data = np.array(force_nn_data)
print("\nRemove the zeroth time step from the data.")
displacement_nn_data = displacement_nn_data[:, 1:, :, :]
force_nn_data = force_nn_data[:, 1:]
I1bar_nn_data = np.array(I1bar_nn_data)
I2bar32_nn_data = np.array(I2bar32_nn_data)
Ibar_nn_data = np.stack((I1bar_nn_data, I2bar32_nn_data), axis=-1)
Wbar_nn_data = np.array(Wbar_nn_data)
connectivity = data["connectivity"]
coordinates = data["coordinates"]
graph = no.data_utils.mesh_to_graph(connectivity)


# DISCPLACEMENT ENCODING WITH LAPLACIAN EIGENFUNCTIONS
Laplace_data = np.load("data/" + experiment_name + "/Laplace_encoding_" + str(n_eigen) + ".npz", allow_pickle=True)
nodes_left = Laplace_data["nodes_left"]
nodes_right = Laplace_data["nodes_right"]
nodes_bottom = Laplace_data["nodes_bottom"]
nodes_top = Laplace_data["nodes_top"]
n_eigen = Laplace_data["n_eigen"]
K, M = assemble_K_M(coordinates, connectivity)
eigenvecs_u1 = Laplace_data["eigenvecs_u1"]
eigenvecs_u2 = Laplace_data["eigenvecs_u2"]
u1_lift = Laplace_data["u1_lift"]
u2_lift = Laplace_data["u2_lift"]

disp_encode_nn_data = np.zeros((displacement_nn_data.shape[0], displacement_nn_data.shape[1], n_eigen, displacement_nn_data.shape[3]))

for i in range(displacement_nn_data.shape[0]): # loop over simulations
    for j in range(displacement_nn_data.shape[1]): # loop over time steps
        u1 = displacement_nn_data[i, j, :, 0]
        a1 = eigenvecs_u1.T @ (M @ u1)
        disp_encode_nn_data[i, j, :, 0] = a1

        u2 = displacement_nn_data[i, j, :, 1]
        w2 = u2 - np.min(u2[nodes_bottom]) * u2_lift
        a2 = eigenvecs_u2.T @ (M @ w2)
        disp_encode_nn_data[i, j, :, 1] = a2


# VERBOSE
print("\nData:")
print("Number of simulations:", len(simulation_list))
print("Number of simulations loaded:", displacement_nn_data.shape[0])
print("Number of simulations ignored:", n_ignored)
print("Number of time steps:", displacement_nn_data.shape[1])
print("Number of spatial points:", displacement_nn_data.shape[2])
print("Number of dimensions:", displacement_nn_data.shape[3])
print("Number of invariants sampled:", Ibar_nn_data.shape[2])
print("displacement_nn_data shape:", displacement_nn_data.shape)
print("disp_encode_nn_data shape:", disp_encode_nn_data.shape)
print("force_nn_data shape:", force_nn_data.shape)
print("Ibar_nn_data shape:", Ibar_nn_data.shape)
print("Wbar_nn_data shape:", Wbar_nn_data.shape)
print("\nMesh:")
print("Number of spatial points in the mesh:", coordinates.shape[0])
print("Number of spatial dimensions in the mesh:", coordinates.shape[1])
print("Number of elements in the mesh:", connectivity.shape[0])
print("Number of nodes per element in the mesh:", connectivity.shape[1])
print("Number of edges in the graph:", graph.shape[0])
print("Number of vertices per edge in the graph:", graph.shape[1])


# SPLIT TRAINING, VALIDATION, AND TEST DATA
n_data = displacement_nn_data.shape[0]
i_train = int(0.8 * n_data)
i_val = int(0.1 * n_data)
idx = torch.randperm(n_data)
idx_train = idx[:i_train]
idx_val = idx[i_train:i_train + i_val]
idx_test = idx[i_train + i_val:]
print("Train indices:", idx_train.numpy()[:5], "...")
print("Validation indices:", idx_val.numpy()[:5], "...")
print("Test indices:", idx_test.numpy()[:5], "...")


# SAVE DATA
# make directory if it does not exist
if not os.path.exists("data/" + experiment_name): os.makedirs("data/" + experiment_name)
np.savez(
        "data/" + experiment_name + "/" + training_name + ".npz",
        simulation_list=simulation_list,
        displacement_nn_data=displacement_nn_data,
        disp_encode_nn_data=disp_encode_nn_data,
        force_nn_data=force_nn_data,
        Ibar_nn_data=Ibar_nn_data,
        Wbar_nn_data=Wbar_nn_data,
        coordinates=coordinates,
        connectivity=connectivity,
        graph=graph,
        n_eigen=n_eigen,
        Ibar_sampling_name=Ibar_sampling_name,
        Ibar_range=Ibar_range,
        n_Ibar_sampling=n_Ibar_sampling,
        idx_train=idx_train.numpy(),
        idx_val=idx_val.numpy(),
        idx_test=idx_test.numpy(),
    )




















