import basix
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import ufl

# dolfinx and fenicsx imports
import dolfinx
from dolfinx import fem, io, log, nls, plot
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmshio
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc

default_scalar_type = PETSc.ScalarType

def run_simulation(Wbar,disp_max=-1.0,n_steps=10,debug=False):
    start = time.time()

    # Simulation settings
    if debug:
        disp_max = -0.01
        n_steps = 1
    else:
        disp_max = disp_max
        n_steps = n_steps
    mesh_file = "fenicsx/gmsh/plate_with_corner_hole_3D.msh"

    # MESH
    mesh, cell_markers, facet_markers = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, gdim=3)
    metadata = {"quadrature_degree": 9}

    mesh_back = np.load("fenicsx/gmsh/plate_with_corner_hole_3D_back.npz", allow_pickle=False)
    coordinates_back = mesh_back["coordinates"]
    connectivity_back = mesh_back["connectivity"]

    # preprocess points
    all_points = [coordinates_back[i,:].reshape(1,3) for i in range(coordinates_back.shape[0])]
    points = np.vstack(all_points)
    tree = dolfinx.geometry.bb_tree(mesh, 3)
    point_results = []
    for point in all_points:
        cells = []
        points_on_proc = []
        cell_candidates = dolfinx.geometry.compute_collisions_points(tree, point)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(mesh, cell_candidates, point)  # Actual cells
        # Check if point is inside mesh.
        for i, p in enumerate(point):
            if len(colliding_cells.links(i)) > 0:
                points_on_proc.append(p)
                cells.append(colliding_cells.links(i)[0])
        point_results.append((np.array(points_on_proc, dtype=np.float64), cells))

    # FUNCTION SPACES
    P1 = basix.ufl.element("CG", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P2 = basix.ufl.element("CG", mesh.basix_cell(), 1)
    el_mixed = basix.ufl.mixed_element([P1, P2])
    W = dolfinx.fem.functionspace(mesh, el_mixed)
    VV = dolfinx.fem.functionspace(mesh,('CG', 1, (mesh.geometry.dim, ))) #To eventually interpolate 1-order tensors (vector)
    SS = dolfinx.fem.functionspace(mesh, ("DG", 0)) #To interpolate scalar fields

    # FUNCTIONS IN THE FUNCTION SPACE V (Function - Trial - Test)
    w = dolfinx.fem.Function(W)
    (u, p) = ufl.split(w)
    dw = ufl.TrialFunction(W)
    (v_u, v_p) = ufl.TestFunctions(W)

    # BOUNDARY CONDITIONS
    tol=1e-4
    def bnd_left(x):
        return np.isclose(x[0],0.0,tol)
    def bnd_right(x):
        return np.isclose(x[0],1.0,tol)
    def bnd_bottom(x):
        return np.isclose(x[1],0.0,tol)
    def bnd_top(x):
        return np.isclose(x[1],1.0,tol)
    def bnd_back(x):
        return np.isclose(x[2],0.0,tol)

    fdim = mesh.topology.dim - 1
    facets_bottom = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bnd_bottom)
    facets_right = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bnd_right)
    facets_top = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bnd_top)
    facets_back = dolfinx.mesh.locate_entities_boundary(mesh, fdim, bnd_back)

    facets_marked = np.hstack([
        facets_bottom,
        facets_right,
        facets_top,
        facets_back,
        ])
    TAG_BOTTOM = 1
    TAG_RIGHT = 2
    TAG_TOP = 3
    TAG_BACK = 4
    values_marked = np.hstack([
        np.full_like(facets_bottom, TAG_BOTTOM),
        np.full_like(facets_right, TAG_RIGHT),
        np.full_like(facets_top, TAG_TOP),
        np.full_like(facets_back, TAG_BACK),
        ])
    facets_sorted = np.argsort(facets_marked)
    facets_meshtags = dolfinx.mesh.meshtags(mesh, fdim, facets_marked[facets_sorted], values_marked[facets_sorted])

    dofs_bottom_x = fem.locate_dofs_topological(W.sub(0).sub(0), facets_meshtags.dim, facets_meshtags.find(TAG_BOTTOM))
    dofs_bottom_y = fem.locate_dofs_topological(W.sub(0).sub(1), facets_meshtags.dim, facets_meshtags.find(TAG_BOTTOM))
    dofs_bottom_z = fem.locate_dofs_topological(W.sub(0).sub(2), facets_meshtags.dim, facets_meshtags.find(TAG_BOTTOM))
    dofs_right_x = fem.locate_dofs_topological(W.sub(0).sub(0), facets_meshtags.dim, facets_meshtags.find(TAG_RIGHT))
    dofs_top_y = fem.locate_dofs_topological(W.sub(0).sub(1), facets_meshtags.dim, facets_meshtags.find(TAG_TOP))
    dofs_back_z = fem.locate_dofs_topological(W.sub(0).sub(2), facets_meshtags.dim, facets_meshtags.find(TAG_BACK))

    # dofs_back = fem.locate_dofs_topological(VV, facets_meshtags.dim, facets_meshtags.find(TAG_BACK))
    # coords = VV.tabulate_dof_coordinates().reshape((-1, mesh.geometry.dim))
    # coords_back = coords[dofs_back][:,:2] # only x and y coordinates, since z is zero for the back face

    disp_0 = dolfinx.fem.Constant(mesh, 0.0)
    disp_imposed = dolfinx.fem.Constant(mesh, 0.0)
    bcs = [
                fem.dirichletbc(disp_0, dofs_bottom_x, W.sub(0).sub(0)),
                fem.dirichletbc(disp_imposed, dofs_bottom_y, W.sub(0).sub(1)),
                # fem.dirichletbc(disp_0, dofs_bottom_z, W.sub(0).sub(2)),
                fem.dirichletbc(disp_0, dofs_top_y, W.sub(0).sub(1)),
                fem.dirichletbc(disp_0, dofs_right_x, W.sub(0).sub(0)),
                fem.dirichletbc(disp_0, dofs_back_z, W.sub(0).sub(2)),
                ]
    
    # PRIMARY FIELDS
    dd = len(u) # Spatial dimension
    I = ufl.Identity(dd) # Identity tensor
    F = ufl.variable(I + ufl.grad(u)) # Deformation gradient from current time step - IMPORTANT: make F a ufl variable to be able to differentiate it

    # CONSTITUTIVE MODEL
    def Piso(F):
        CG = ufl.dot(F.T,F)
        _I1 = ufl.tr(CG)
        _I2 = (1.0/2.0) * (ufl.tr(CG) ** 2 - ufl.tr(CG * CG))
        _J = ufl.det(F)
        _I1bar = _J**(-2.0/3.0) * _I1
        _I2bar = _J**(-4.0/3.0) * _I2
        _I2bar32 = _I2bar**(3.0/2.0)
        _Wbar = Wbar(_I1bar,_I2bar32)
        return ufl.diff(_Wbar, F)

    def Pvol(F,p):
        return -p*ufl.det(F)*ufl.inv(F).T

    def Ptotal(F,p):
        return Piso(F)+Pvol(F,p)

    # VARIATIONAL PROBLEM - WEAK FORM
    dx = ufl.Measure('dx')(domain=mesh, subdomain_data=cell_markers, metadata=metadata) #Integration measures for each subdomain
    # ds = ufl.Measure("ds", domain=mesh, subdomain_data=facets_meshtags, metadata=metadata)
    Res_u = (ufl.inner(Ptotal(F,p), ufl.grad(v_u)) - v_p*(ufl.det(F)-1)) * dx
    # _normal = ufl.FacetNormal(mesh)
    # _traction = ufl.dot(Ptotal(F,p),_normal)

    # SET UP NON LINEAR VARIATIONAL PROBLEM
    problem = fem.petsc.NonlinearProblem(Res_u, w, bcs, ufl.derivative(Res_u, w, dw))
    solver = nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.atol = 1e-8
    solver.rtol = 1e-8
    solver.convergence_criterion = "incremental"

    # SETTING UP INTERPOLATION
    u_expr = dolfinx.fem.Expression(u, VV.element.interpolation_points())
    u_func = dolfinx.fem.Function(VV, name = "u")
    J_expr = dolfinx.fem.Expression(ufl.det(F), SS.element.interpolation_points())
    J_func = dolfinx.fem.Function(SS, name = "J")
    volume = fem.assemble_scalar(fem.form(1.0 * dx))

    # DATA ARRAYS
    coordinates = coordinates_back[:,:2]
    connectivity = connectivity_back
    displacement_data = np.zeros((n_steps+1, coordinates_back.shape[0], 2))
    force_data = np.zeros((n_steps+1,))

    # SIMULATION
    end = time.time()
    print(f"Time to setup simulation: {end - start:.6f} seconds")
    try:
        # log.set_log_level(log.LogLevel.INFO)
        for i in range(n_steps + 1):
            print()
            start = time.time()
            print("Step: ", i, "/", n_steps)
            disp_imposed.value = disp_max * i/n_steps
            print(f"Current imposed displacement: {disp_imposed.value:.6f}")
            num_its, converged = solver.solve(w)
            if converged:
                print("Newton solver converged in ", num_its, " iterations.")
            else:
                raise ValueError("Newton solver did not converge.")
            w.x.scatter_forward()
            u_func.interpolate(u_expr)

            for idx_point, (points_on_proc, cells) in enumerate(point_results):
                if len(points_on_proc) > 0:
                    disp = u_func.eval(points_on_proc, cells)
                    displacement_data[i, idx_point, :] = disp[:2]
            
            R = fem.petsc.assemble_vector(fem.form(Res_u))
            R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            force_data[i] = 0.0
            for dof in dofs_bottom_y: force_data[i] += R.array[dof]
            # force_data[i] = fem.assemble_scalar(fem.form(_traction[1] * ds(TAG_BOTTOM)))
            J_func.interpolate(J_expr)
            J_integral = fem.assemble_scalar(fem.form(J_func * dx))

            end = time.time()
            print(f"Time for {i}. step: {end - start:.6f} seconds")
            print(f"Reaction force: {force_data[i]:.6f}")
            print(f"Mean value of J over domain: {J_integral/volume:.6f}")
        converged = True
    except:
        converged = False
        print("Simulation failed or did not converge.")

    return converged, coordinates, connectivity, displacement_data, force_data










