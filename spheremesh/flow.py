import igl
import numpy as np
import scipy as sp 
from .fit import normalize_area

from scipy.sparse.linalg import spsolve
from scipy.special import sph_harm
from numpy.linalg import norm



def normalize(vertices):
	centroid = np.mean(vertices, axis=0);
	vertices -= centroid;
	radii = norm(vertices, axis=1);

	m = np.amax(radii)

	vertices /= m;
	return vertices;

def get_error(vertices):
	#todo get better error, like overlapping faces
	centroid = np.mean(vertices, axis=0);
	vertices -= centroid;
	radii = norm(vertices, axis=1);

	m = np.amax(radii)
	err = (m - np.amin(radii))/m

	return err;

def project_sphere(v):
	return np.divide(v, norm(v, axis=1).reshape((-1,1)));


def get_flipped_normals(vertices, faces):
	normals = igl.per_vertex_normals(vertices, faces)
	dot = np.sum(normals*vertices,axis=1)
	return sum(dot<0)/vertices.shape[0]



def conformal_flow(vertices, faces):
	vertices = normalize(np.copy(vertices))

	L = igl.cotmatrix(vertices, faces)
	

	itrs = 20
	time_step = .1

	for i in range(itrs):
		vertices = normalize(vertices)
		
		# this is a lumped mass matrix
		M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)
		S = (M - time_step * L)
		b = M.dot(vertices)
		vertices = spsolve(S, b)


	vertices = normalize(vertices)
	
	print("flow error: " + str(get_error(vertices)))

	vertices = project_sphere(vertices)

	print("percentage of flipped faces: " + str(100*get_flipped_normals(vertices, faces)))
	return vertices


#
# Moebius centering
# 

def compute_jacobian(M,vertices):
	areas = M.diagonal()
	J = np.zeros((3,3))
	for i in range(vertices.shape[0]):
		J += areas[i]*(np.eye(3) - np.outer(vertices[i],vertices[i]))

	J *= 2
	return J

def mobius_center(orig_v,vertices,faces):
	M = igl.massmatrix(orig_v, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)

	for i in range(10):
		# center of mass
		mu = np.sum(M.dot(vertices), axis=0)

		err = norm(mu)
		print("mobius error: " + str(norm(mu)))
		if err < 1e-10:
			break

		# c = -J^-1 * mu
		J = compute_jacobian(M,vertices)
		c = -np.linalg.inv(J).dot(mu)

		# compute inversion
		vertices = np.divide((vertices + c), norm(vertices + c, axis=1).reshape((-1,1))**2);
		vertices = (1 - norm(c)**2)*vertices + c

	return vertices

def get_rho(orig_v,vertices,faces):
	orig_v = normalize_area(orig_v, faces)
	a1 = igl.massmatrix(orig_v, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
	a2 = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC).diagonal()
	
	return np.divide(a1,a2)

