import spheremesh as sh
import igl
import numpy as np
from numpy.linalg import norm


filename = "data/cell.obj"
l_max = 24

orig_v, faces = igl.read_triangle_mesh(filename)

sphere_verts = sh.conformal_flow(orig_v, faces)
sphere_verts = sh.mobius_center(orig_v,sphere_verts,faces)

orig_v, sphere_verts = sh.canonical_rotation(orig_v, sphere_verts, faces)

weights, Y_mat = sh.IRF(orig_v, sphere_verts, faces, l_max)

reconstruct = Y_mat.dot(weights)

p = sh.plot.new()
sh.plot.mesh(p, orig_v, faces)

def set_max(value):
	res = int(value)**2
	res2 = int(value+1)**2
	ww = np.zeros(weights.shape[0])

	ww[0:res] = 1
	ww[res:res2] = 1

	ww = np.vstack((ww,ww,ww))
	reconstruct = Y_mat.dot((weights)*ww.T )

	err = reconstruct - orig_v

	err = norm(err, axis=1)

	sh.plot.mesh(p, reconstruct, faces)

	print("mean reconstruction error: " + str(np.mean(err)))
	return


p.add_slider_widget(set_max, [1, l_max], title='max degree')

p.show()