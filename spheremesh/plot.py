import pyvista as pv
import numpy as np

def plot(V,F,K=None):
	p = pv.Plotter()
	n,m = F.shape 
	threes = np.full((n,1),3)
	face_arr = np.hstack((threes,F)).flatten()
	surf = pv.PolyData(V, face_arr)
	if K is not None:
		surf["colors"] = K
	#p.add_mesh(surf, name='mesh', show_edges=False, rgb=True, smooth_shading=True)
	p.add_mesh(surf, name='mesh', show_edges=True, rgb=False, smooth_shading=False)
	p.show_axes()
	p.show()
