# external imports
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix

def assemble_K_M(coordinates, connectivity):
    """
    Assemble FEM stiffness and mass matrices for a 2D triangular mesh.

    Parameters
    ----------
    coordinates : (N,2) ndarray
    connectivity : (T,3) ndarray

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Stiffness matrix.
    M : scipy.sparse.csr_matrix
        Mass matrix.
    """

    n_nodes = coordinates.shape[0]

    rows_K = []
    cols_K = []
    data_K = []

    rows_M = []
    cols_M = []
    data_M = []

    for tri in connectivity:
        nodes = coordinates[tri]

        x1, y1 = nodes[0]
        x2, y2 = nodes[1]
        x3, y3 = nodes[2]

        # Jacobian determinant
        detJ = (
            (x2 - x1) * (y3 - y1)
            - (x3 - x1) * (y2 - y1)
        )

        area = 0.5 * abs(detJ)

        if area <= 0:
            raise ValueError("Degenerate triangle encountered.")

        # Gradients of barycentric basis functions
        b = np.array([
            y2 - y3,
            y3 - y1,
            y1 - y2
        ])

        c = np.array([
            x3 - x2,
            x1 - x3,
            x2 - x1
        ])

        # Local stiffness matrix
        Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)

        # Consistent local mass matrix
        Me = (area / 12.0) * np.array([
            [2, 1, 1],
            [1, 2, 1],
            [1, 1, 2]
        ])

        # Assembly
        for i_local in range(3):
            i_global = tri[i_local]

            for j_local in range(3):
                j_global = tri[j_local]

                rows_K.append(i_global)
                cols_K.append(j_global)
                data_K.append(Ke[i_local, j_local])

                rows_M.append(i_global)
                cols_M.append(j_global)
                data_M.append(Me[i_local, j_local])

    K = coo_matrix(
        (data_K, (rows_K, cols_K)),
        shape=(n_nodes, n_nodes)
    ).tocsr()

    M = coo_matrix(
        (data_M, (rows_M, cols_M)),
        shape=(n_nodes, n_nodes)
    ).tocsr()

    return K, M

def evaluate_basis_at_points(points,
                             coordinates,
                             connectivity,
                             eigenvecs):
    """
    Evaluate Laplace eigenfunctions at arbitrary points.

    Parameters
    ----------
    points : (n_pts,2)
    coordinates : (n_nodes,2)
    connectivity : (n_elem,3)
    eigenvecs : (n_nodes,n_eigen)

    Returns
    -------
    Phi : (n_pts,n_eigen)
        Basis evaluated at points.
    """

    n_pts = points.shape[0]
    n_eigen = eigenvecs.shape[1]
    Phi = np.zeros((n_pts, n_eigen))

    for p, x in enumerate(points):

        found = False

        for elem in connectivity:

            verts = coordinates[elem]

            A = np.array([
                [verts[0,0], verts[1,0], verts[2,0]],
                [verts[0,1], verts[1,1], verts[2,1]],
                [1.0,        1.0,        1.0]
            ])

            rhs = np.array([x[0], x[1], 1.0])

            lam = np.linalg.solve(A, rhs)

            if np.all(lam >= -1e-12):

                phi_nodes = eigenvecs[elem, :]  # (3,n_eigen)

                Phi[p, :] = lam @ phi_nodes
                found = True
                break

        if not found:
            raise ValueError(
                f"Point {x} lies outside the mesh."
            )

    return Phi

