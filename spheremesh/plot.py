import pyvista as pv
import numpy as np

def new():
	return pv.Plotter()

def mesh(p, V,F,K=None):
	n,m = F.shape 
	threes = np.full((n,1),3)
	face_arr = np.hstack((threes,F)).flatten()
	surf = pv.PolyData(V, face_arr)
	if K is not None:
		surf["colors"] = K
		p.add_mesh(surf, name='mesh', show_edges=False, rgb=False, smooth_shading=True, cmap = "bwr", clim = [-2,2])
	else:
		p.add_mesh(surf, name='mesh', show_edges=False, rgb=False, smooth_shading=False)

	p.show_axes()
	#p.show()
	return p
