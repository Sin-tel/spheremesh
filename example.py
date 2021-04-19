import spheremesh as sh
import igl
import numpy as np


filename = "cell.obj"
l_max = 16

# read the example .obj triangle mesh
# this gives us an array of vertex positions and list of faces (triangles), which are triples of vertex indices
orig_v, faces = igl.read_triangle_mesh(filename)

# run the conformal flow to get sphere map
sphere_verts = sh.flow.conformal_flow(orig_v, faces)
# moebius centering algorithm to find optimal sphere map
sphere_verts = sh.flow.mobius_center(orig_v,sphere_verts,faces)

# find a canonical orientation via first order ellipoid method
orig_v, sphere_verts = sh.fit.canonical_rotation(orig_v, sphere_verts, faces)

# run iterative residual fitting to get the spherical harmonics decomposition, up to degree l_max
# note that you can just pass it the original vertex positions as one array. no reason to run it seperately for each coordinate
weights, Y_mat = sh.fit.IRF(orig_v, sphere_verts, faces, l_max)

#recover = Y_mat.dot(weights)


np.set_printoptions(precision=4, suppress = True, floatmode = "fixed")
print(weights[0:9])


#sh.plot(recover, faces)