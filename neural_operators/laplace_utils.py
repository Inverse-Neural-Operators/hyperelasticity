# external imports
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

# costum imports
from neural_operators.fem_utils import assemble_K_M, evaluate_basis_at_points

disp_max = -2.0

def Laplace_encoding(Laplace_data, displacement_data):
    """
    Map displacement data to Laplace encoding coefficients.
    Displacement data is expected to be given at the nodes of the FEM mesh used for Laplace eigenvalue problem.

    """

    # load Laplace data
    coordinates = Laplace_data["coordinates"]
    connectivity = Laplace_data["connectivity"]
    nodes_left = Laplace_data["nodes_left"]
    nodes_right = Laplace_data["nodes_right"]
    nodes_bottom = Laplace_data["nodes_bottom"]
    nodes_top = Laplace_data["nodes_top"]
    n_eigen = Laplace_data["n_eigen"]
    eigenvecs_u1 = Laplace_data["eigenvecs_u1"]
    eigenvecs_u2 = Laplace_data["eigenvecs_u2"]
    u1_lift = Laplace_data["u1_lift"]
    u2_lift = Laplace_data["u2_lift"]

    K, M = assemble_K_M(coordinates, connectivity)

    # Laplace encoding
    disp_encode_data = np.zeros((1, displacement_data.shape[0], n_eigen, displacement_data.shape[2]))
    for j in range(displacement_data.shape[0]): # loop over time steps
        u1 = displacement_data[j, :, 0]
        a1 = eigenvecs_u1.T @ (M @ u1)
        disp_encode_data[0, j, :, 0] = a1

        u2 = displacement_data[j, :, 1]
        w2 = u2 - np.min(u2[nodes_bottom]) * u2_lift
        a2 = eigenvecs_u2.T @ (M @ w2)
        disp_encode_data[0, j, :, 1] = a2

    return disp_encode_data

def Laplace_encoding_scarce(Laplace_data, displacement_data, scarce):
    """
    Map displacement data to Laplace encoding coefficients.
    Displacement data is expected to be given at the nodes of the FEM mesh used for Laplace eigenvalue problem, but only at a subset of the nodes (scarce data).

    """

    # load Laplace data
    coordinates = Laplace_data["coordinates"]
    connectivity = Laplace_data["connectivity"]
    nodes_left = Laplace_data["nodes_left"]
    nodes_right = Laplace_data["nodes_right"]
    nodes_bottom = Laplace_data["nodes_bottom"]
    nodes_top = Laplace_data["nodes_top"]
    n_eigen = Laplace_data["n_eigen"]
    eigenvecs_u1 = Laplace_data["eigenvecs_u1"]
    eigenvecs_u2 = Laplace_data["eigenvecs_u2"]
    u1_lift = Laplace_data["u1_lift"]
    u2_lift = Laplace_data["u2_lift"]

    idx_scarce = np.random.choice(coordinates.shape[0], size=scarce, replace=False)
    idx_scarce = np.sort(idx_scarce)
    eigenvecs_u1_scarce = eigenvecs_u1[idx_scarce, :]
    eigenvecs_u2_scarce = eigenvecs_u2[idx_scarce, :]

    # Laplace encoding from scarce data
    n_step = displacement_data.shape[0]
    delta_u2 = disp_max / n_step
    disp_encode_data = np.zeros((1, n_step, n_eigen, displacement_data.shape[2]))
    for j in range(n_step): # loop over time steps
        u1_scarce = displacement_data[j, idx_scarce, 0]
        u2_scarce = displacement_data[j, idx_scarce, 1]
        w2_scarce = u2_scarce - (j+1) * delta_u2 * u2_lift[idx_scarce]
        a1_scarce, *_ = np.linalg.lstsq(eigenvecs_u1_scarce, u1_scarce, rcond=None)
        a2_scarce, *_ = np.linalg.lstsq(eigenvecs_u2_scarce, w2_scarce, rcond=None)
        disp_encode_data[0, j, :, 0] = a1_scarce
        disp_encode_data[0, j, :, 1] = a2_scarce

    return disp_encode_data, idx_scarce

def Laplace_encoding_scattered(Laplace_data, displacement_data_scattered, coordinates_scattered):
    """
    Map displacement data to Laplace encoding coefficients.
    Displacement data is expected to be given at scattered points, not necessarily at the nodes of the FEM mesh used for Laplace eigenvalue problem.

    """

    # load Laplace data
    coordinates = Laplace_data["coordinates"]
    connectivity = Laplace_data["connectivity"]
    nodes_left = Laplace_data["nodes_left"]
    nodes_right = Laplace_data["nodes_right"]
    nodes_bottom = Laplace_data["nodes_bottom"]
    nodes_top = Laplace_data["nodes_top"]
    n_eigen = Laplace_data["n_eigen"]
    eigenvecs_u1 = Laplace_data["eigenvecs_u1"]
    eigenvecs_u2 = Laplace_data["eigenvecs_u2"]
    u1_lift = np.zeros_like(coordinates_scattered[:, 0])
    u2_lift = 1.0 - coordinates_scattered[:,1]

    eigenvecs_u1_scattered = evaluate_basis_at_points(
        coordinates_scattered,
        coordinates,
        connectivity,
        eigenvecs_u1
    )
    eigenvecs_u2_scattered = evaluate_basis_at_points(
        coordinates_scattered,
        coordinates,
        connectivity,
        eigenvecs_u2
    )

    # Laplace encoding from scarce data
    n_step = displacement_data_scattered.shape[0]
    delta_u2 = disp_max / n_step
    disp_encode_data = np.zeros((1, n_step, n_eigen, displacement_data_scattered.shape[2]))
    for j in range(n_step): # loop over time steps
        u1 = displacement_data_scattered[j, :, 0]
        u2 = displacement_data_scattered[j, :, 1]
        w2 = u2 - (j+1) * delta_u2 * u2_lift
        a1_scattered, *_ = np.linalg.lstsq(eigenvecs_u1_scattered, u1, rcond=None)
        a2_scattered, *_ = np.linalg.lstsq(eigenvecs_u2_scattered, w2, rcond=None)
        disp_encode_data[0, j, :, 0] = a1_scattered
        disp_encode_data[0, j, :, 1] = a2_scattered

    return disp_encode_data
