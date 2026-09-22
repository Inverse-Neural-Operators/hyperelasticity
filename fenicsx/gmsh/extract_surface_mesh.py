import meshio
import numpy as np

mesh_name = "plate_with_corner_hole_3D"

mesh = meshio.read("fenicsx/gmsh/" + mesh_name + ".msh")

triangles = mesh.cells_dict["triangle"]
tri_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]

connectivity_front = triangles[tri_tags == mesh.field_data["front"][0]]
connectivity_back = triangles[tri_tags == mesh.field_data["back"][0]]

unique_nodes_back = np.unique(connectivity_back)
node_map = {old: new for new, old in enumerate(unique_nodes_back)}
connectivity_back_local = np.vectorize(node_map.get)(connectivity_back)

np.savez(
    "fenicsx/gmsh/" + mesh_name + "_back.npz",
    coordinates = mesh.points[unique_nodes_back],
    connectivity = connectivity_back_local,
)


