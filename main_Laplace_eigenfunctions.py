# external imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.tri as mtri
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix
import seaborn as sns

rs = 42
np.random.seed(rs)

# custom imports
import neural_operators as no
from neural_operators.fem_utils import assemble_K_M

# plotting settings
paper_style = True
if paper_style:
    # plt.style.use('media/paper_style.mplstyle')
    plt.style.use('media/paper_style_large_font.mplstyle')

# settings
experiment_name = "plate_with_corner_hole_3D_u2"
n_eigen = 100
save = True
show = False

# # load mesh
# mesh = np.load("fenicsx/gmsh/plate_with_corner_hole_3D_back.npz", allow_pickle=False)
# coordinates = mesh["coordinates"]
# connectivity = mesh["connectivity"]

# load data
data = np.load(
    "fenicsx/data/" + experiment_name + "/NeoHooke.npz",
    allow_pickle=False
    )
coordinates = data["coordinates"]
connectivity = data["connectivity"]
displacement_data = data["displacement_data"]
force_data = data["force_data"]

print("coordinates shape:", coordinates.shape)
print("connectivity shape:", connectivity.shape)
print("displacement_data shape:", displacement_data.shape)
print("force_data shape:", force_data.shape)

triang = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1], connectivity)

# # PLOT MESH
# no.plot_mesh(coordinates, connectivity)
# if show: plt.show()

# # PLOT DISPLACEMENT
# no.plot_scalar_field(coordinates, connectivity, displacement_data[-1, :, 0], label=r"$u_1$")
# plt.title(r"Displacement $u_1$")
# if show: plt.show()
# no.plot_scalar_field(coordinates, connectivity, displacement_data[-1, :, 1], label=r"$u_2$")
# plt.title(r"Displacement $u_2$")
# if show: plt.show()

# LAPLACIAN EIGENFUNCTIONS
K, M = assemble_K_M(coordinates, connectivity)

# BOUNDARY CONDITIONS
tol = 1e-6
nodes_left = np.where(np.abs(coordinates[:,0] - 0.0) < tol)[0]
nodes_right = np.where(np.abs(coordinates[:,0] - 1.0) < tol)[0]
nodes_bottom = np.where(np.abs(coordinates[:,1] - 0.0) < tol)[0]
nodes_top = np.where(np.abs(coordinates[:,1] - 1.0) < tol)[0]
nodes_dirichlet_u1 = np.union1d(nodes_right, nodes_bottom)
nodes_free_u1 = np.setdiff1d(np.arange(coordinates.shape[0]), nodes_dirichlet_u1)
nodes_dirichlet_u2 = np.union1d(nodes_top, nodes_bottom)
nodes_free_u2 = np.setdiff1d(np.arange(coordinates.shape[0]), nodes_dirichlet_u2)

K_free_u1 = K[nodes_free_u1][:, nodes_free_u1]
M_free_u1 = M[nodes_free_u1][:, nodes_free_u1]
K_free_u2 = K[nodes_free_u2][:, nodes_free_u2]
M_free_u2 = M[nodes_free_u2][:, nodes_free_u2]

eigenvals_u1, eigenvecs_free_u1 = eigsh(
    K_free_u1,
    k=n_eigen,
    M=M_free_u1,
    sigma=0.0,
    v0 = np.ones(K_free_u1.shape[0])
)
eigenvals_u2, eigenvecs_free_u2 = eigsh(
    K_free_u2,
    k=n_eigen,
    M=M_free_u2,
    sigma=0.0,
    v0 = np.ones(K_free_u2.shape[0])
)

# sign convention for reproducibility
for i in range(n_eigen):
    if eigenvecs_free_u1[np.argmax(np.abs(eigenvecs_free_u1[:, i])), i] < 0:
        eigenvecs_free_u1[:, i] *= -1
for i in range(n_eigen):
    if eigenvecs_free_u2[np.argmax(np.abs(eigenvecs_free_u2[:, i])), i] < 0:
        eigenvecs_free_u2[:, i] *= -1

eigenvecs_u1 = np.zeros((coordinates.shape[0], n_eigen))
eigenvecs_u2 = np.zeros((coordinates.shape[0], n_eigen))
eigenvecs_u1[nodes_free_u1, :] = eigenvecs_free_u1
eigenvecs_u2[nodes_free_u2, :] = eigenvecs_free_u2

# PLOT LAPLACIAN EIGENFUNCTIONS
# for i in range(1):
for i in range(6):
    no.plot_scalar_field(coordinates, connectivity, eigenvecs_u1[:, i], label=fr"$\phi_{{1{i+1}}}(\boldsymbol{{X}})$")
    # plt.title(fr"Eigenfunction $\phi_{i}$ (u1)")
    if save: plt.savefig(f"media/images/paper/eigenfunction_u1_{i}.png", transparent=True, bbox_inches="tight")
    if show: plt.show()
    no.plot_scalar_field(coordinates, connectivity, eigenvecs_u2[:, i], label=fr"$\phi_{{2{i+1}}}(\boldsymbol{{X}})$")
    # plt.title(fr"Eigenfunction $\phi_{i}$ (u2)")
    if save: plt.savefig(f"media/images/paper/eigenfunction_u2_{i}.png", transparent=True, bbox_inches="tight")
    if show: plt.show()

# ENCODING
u1 = displacement_data[-1, :, 0]
u1_lift = np.zeros_like(u1)
a1 = eigenvecs_u1.T @ (M @ u1)

u2 = displacement_data[-1, :, 1]
u2_lift = 1.0 - coordinates[:,1]
w2 = u2 - np.max(u2[nodes_bottom]) * u2_lift
a2 = eigenvecs_u2.T @ (M @ w2)

# DECODING
u1_approx = eigenvecs_u1 @ a1
w2_approx = eigenvecs_u2 @ a2
u2_approx = w2_approx + np.max(u2[nodes_bottom]) * u2_lift

print("Error u1:", np.linalg.norm(u1 - u1_approx) / np.linalg.norm(u1))
print("Error u2:", np.linalg.norm(u2 - u2_approx) / np.linalg.norm(u2))

# # PLOT DISPLACEMENT
# no.plot_scalar_field(coordinates, connectivity, u1_approx, label=r"$u_1$ (approx)")
# plt.title(r"Displacement $u_1$ (approx)")
# if show: plt.show()
# no.plot_scalar_field(coordinates, connectivity, u2_approx, label=r"$u_2$ (approx)")
# plt.title(r"Displacement $u_2$ (approx)")
# if show: plt.show()

# # PLOT DEFORMED MESHES
# coordinates_deformed = coordinates + np.stack((u1, u2), axis=-1)
# coordinates_deformed_approx = coordinates + np.stack((u1_approx, u2_approx), axis=-1)
# no.plot_meshes(coordinates_deformed, connectivity, coordinates_deformed_approx, connectivity)
# if show: plt.show()

np.savez(
        "data/" + experiment_name + "/Laplace_encoding_" + str(n_eigen) + ".npz",
        connectivity=connectivity,
        coordinates=coordinates,
        nodes_left=nodes_left,
        nodes_right=nodes_right,
        nodes_bottom=nodes_bottom,
        nodes_top=nodes_top,
        nodes_dirichlet_u1=nodes_dirichlet_u1,
        nodes_free_u1=nodes_free_u1,
        nodes_dirichlet_u2=nodes_dirichlet_u2,
        nodes_free_u2=nodes_free_u2,
        n_eigen=n_eigen,
        K=K,
        M=M,
        K_free_u1=K_free_u1,
        M_free_u1=M_free_u1,
        K_free_u2=K_free_u2,
        M_free_u2=M_free_u2,
        eigenvecs_u1=eigenvecs_u1,
        eigenvecs_u2=eigenvecs_u2,
        u1_lift=u1_lift,
        u2_lift=u2_lift,
    )

# # CONVERGENCE STUDY
# # n_eigen_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# n_eigen_list = range(1, 51)
# errors_u1 = []
# errors_u2 = []
# for n_eigen in n_eigen_list:
#     eigenvals_u1, eigenvecs_free_u1 = eigsh(
#         K_free_u1,
#         k=n_eigen,
#         M=M_free_u1,
#         sigma=0.0
#     )
#     eigenvals_u2, eigenvecs_free_u2 = eigsh(
#         K_free_u2,
#         k=n_eigen,
#         M=M_free_u2,
#         sigma=0.0
#     )
#     eigenvecs_u1 = np.zeros((coordinates.shape[0], n_eigen))
#     eigenvecs_u2 = np.zeros((coordinates.shape[0], n_eigen))
#     eigenvecs_u1[nodes_free_u1, :] = eigenvecs_free_u1
#     eigenvecs_u2[nodes_free_u2, :] = eigenvecs_free_u2
#     a1 = eigenvecs_u1.T @ (M @ u1)
#     a2 = eigenvecs_u2.T @ (M @ u2)
#     u1_approx = eigenvecs_u1 @ a1
#     u2_approx = eigenvecs_u2 @ a2
#     error_u1 = np.linalg.norm(u1 - u1_approx) / np.linalg.norm(u1)
#     error_u2 = np.linalg.norm(u2 - u2_approx) / np.linalg.norm(u2)
#     errors_u1.append(error_u1)
#     errors_u2.append(error_u2)

# plt.figure()
# plt.plot(n_eigen_list, errors_u1, marker="o", label=r"$u_1$")
# plt.plot(n_eigen_list, errors_u2, marker="o", label=r"$u_2$")
# # plt.yscale("log")
# plt.xlabel("Number of eigenfunctions")
# plt.ylabel("Relative error")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# if show: plt.show()


# interpolate a scarce displacement field
# scarce = 400
# idx_scarce = np.random.choice(coordinates.shape[0], size=scarce, replace=False)
# idx_scarce = np.sort(idx_scarce)

# u1_scarce = u1[idx_scarce]
# u2_scarce = u2[idx_scarce]
# u2_lift = 1.0 - coordinates[:,1]
# w2_scarce = u2[idx_scarce] - np.max(u2[nodes_bottom]) * u2_lift[idx_scarce]

# eigenvecs_u1_scarce = eigenvecs_u1[idx_scarce, :]
# eigenvecs_u2_scarce = eigenvecs_u2[idx_scarce, :]

# a1_scarce, *_ = np.linalg.lstsq(eigenvecs_u1_scarce, u1_scarce, rcond=None)

# a2_scarce, *_ = np.linalg.lstsq(eigenvecs_u2_scarce, w2_scarce, rcond=None)

# u1_interp = eigenvecs_u1 @ a1_scarce
# u2_interp = eigenvecs_u2 @ a2_scarce + np.max(u2[nodes_bottom]) * u2_lift

# print("Error u1:", np.linalg.norm(u1 - u1_interp) / np.linalg.norm(u1))
# print("Error u2:", np.linalg.norm(u2 - u2_interp) / np.linalg.norm(u2))

# # PLOT DEFORMED MESHES
# coordinates_deformed = coordinates + np.stack((u1, u2), axis=-1)
# coordinates_deformed_approx = coordinates + np.stack((u1_interp, u2_interp), axis=-1)
# no.plot_meshes(coordinates_deformed, connectivity, coordinates_deformed_approx, connectivity)
# if show: plt.show()
