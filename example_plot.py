import spheremesh as sh
import igl
import numpy as np


filename = "cell.obj"
l_max = 16

orig_v, faces = igl.read_triangle_mesh(filename)

sphere_verts = sh.flow.conformal_flow(orig_v, faces)
sphere_verts = sh.flow.mobius_center(orig_v,sphere_verts,faces)

orig_v, sphere_verts = sh.fit.canonical_rotation(orig_v, sphere_verts, faces)

weights, Y_mat = sh.fit.IRF(orig_v, sphere_verts, faces, l_max)

# get the reconstructed vertex positions from the spherical harmonics expansion
reconstruct = Y_mat.dot(weights)

sh.plot(reconstruct, faces)