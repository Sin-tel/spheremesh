import spheremesh as sh
import igl
import numpy as np

filename = "data/cell.obj"
l_max = 20

orig_v, faces = igl.read_triangle_mesh(filename)


sphere_verts = sh.conformal_flow(orig_v, faces)
sphere_verts = sh.mobius_center(orig_v,sphere_verts,faces)

orig_v, sphere_verts = sh.canonical_rotation(orig_v, sphere_verts, faces)

weights, Y_mat = sh.IRF(orig_v, sphere_verts, faces, l_max)

reconstruct = Y_mat.dot(weights)

## construct icosphere mesh data
ico_v, ico_f = igl.read_triangle_mesh("spheremesh/icosphere/icosphere.obj")
ico_v = sh.project_sphere(ico_v)
theta = np.arccos(ico_v[:,2])
phi = np.arctan2(ico_v[:,1],ico_v[:,0])

Y_mat2 = []

## make sh matrix
for l in range(0, l_max):
	for m in range(-l,l+1):
		y = sh.sph_real(l, m, phi, theta)
		Y_mat2.append(y)

Y_mat2 = np.vstack(Y_mat2).T

ico_reconstruct = Y_mat2.dot(weights)


## plot
p = sh.plot.new()
sh.plot.mesh(p, reconstruct, faces)

def update(value):
	global sphere_verts, reconstruct, faces, color, p

	if value:
		sh.plot.mesh(p,ico_reconstruct, ico_f)
	else:
		sh.plot.mesh(p,reconstruct, faces)
		
	return

p.add_checkbox_button_widget(update)
p.show()