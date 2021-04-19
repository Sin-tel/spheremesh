import spheremesh as sh
import igl
import numpy as np


filename = "data/cell.obj"
l_max = 32

orig_v, faces = igl.read_triangle_mesh(filename)

sphere_verts = sh.conformal_flow(orig_v, faces)
sphere_verts = sh.mobius_center(orig_v,sphere_verts,faces)

orig_v, sphere_verts = sh.fit.canonical_rotation(orig_v, sphere_verts, faces)

weights, Y_mat = sh.fit.IRF(orig_v, sphere_verts, faces, l_max)

# get the reconstructed vertex positions from the spherical harmonics expansion
reconstruct = Y_mat.dot(weights)

# set rho (area distortion) as the color 
color = np.log(sh.get_rho(orig_v, sphere_verts, faces))

p = sh.plot.new()
sh.plot.mesh(p, orig_v, faces, K = color)

def update(value):
	global sphere_verts, reconstruct, faces, color, p

	if value:
		sh.plot.mesh(p,reconstruct, faces, K = color)
	else:
		sh.plot.mesh(p,sphere_verts, faces, K = color)
		
	return

p.add_checkbox_button_widget(update)
p.show()