import igl
import numpy as np
import scipy as sp 

from scipy.sparse.linalg import spsolve
from scipy.special import sph_harm
from numpy.linalg import norm

def get_area(vertices, faces):
	M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)
	return np.sum(M.dot(np.ones(vertices.shape[0])),axis=0)

# translate mesh so that: origin = center of mass
# scale mesh so that: total area = 4*pi (= area of unit sphere)
def normalize_area(vertices, faces, M = None):
	if M is None:
		M = igl.massmatrix(vertices, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)
	A = get_area(vertices, faces)

	v_w = M.dot(vertices)

	centroid = np.sum(v_w, axis=0) / A
	vertices -= centroid;
	
	vertices /= np.sqrt(A / (np.pi * 4));

	return vertices;


def project_sphere(v):
	return np.divide(v, norm(v, axis=1).reshape((-1,1)));


def sph_real(l, m, phi, theta):
	scale = 1
	if m != 0:
		scale = np.sqrt(2)
	if m >= 0:
		return scale*sph_harm(m, l, phi, theta).real
	else:
		return scale*sph_harm(abs(m), l, phi, theta).imag

# canonical rotation

def canonical_rotation(orig_v, vertices, faces):
	weights, A = IRF(orig_v, vertices, faces, 3)

	Q = np.vstack([-weights[3],-weights[1],weights[2]]).T

	u, s , vh = np.linalg.svd(Q)

	# product here is reversed because the vertices are row vectors (ie: A * v -> v^T * A^T)
	orig_v = orig_v.dot(u)
	vertices = vertices.dot(vh.T)

	# project to sphere again because of numerical issues w rotation
	vertices = project_sphere(vertices)


	# pi rotations: yz, xz, xy
	rx = np.diag([ 1,-1,-1])
	ry = np.diag([-1, 1,-1])
	rz = np.diag([-1,-1, 1])

	M = igl.massmatrix(orig_v, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)


	weights, A = IRF(orig_v, vertices, faces, 8)
	weights_x, A_x = IRF(orig_v.dot(rx), vertices.dot(rx), faces, 8)
	weights_y, A_y = IRF(orig_v.dot(ry), vertices.dot(ry), faces, 8)
	weights_z, A_z = IRF(orig_v.dot(rz), vertices.dot(rz), faces, 8)

	# l = [
	# sum(weights[4])+sum(weights[5])+sum(weights[7]),
	# sum(weights_x[4])+sum(weights_x[5])+sum(weights_x[7]),
	# sum(weights_y[4])+sum(weights_y[5])+sum(weights_y[7]),
	# sum(weights_z[4])+sum(weights_z[5])+sum(weights_z[7]),
	# ]

	l = [
	np.sum(np.sum(M.dot(A.dot(weights)), axis=0)),
	np.sum(np.sum(M.dot(A_x.dot(weights_x)), axis=0)),
	np.sum(np.sum(M.dot(A_y.dot(weights_y)), axis=0)),
	np.sum(np.sum(M.dot(A_z.dot(weights_z)), axis=0)),
	]

	# print(np.sum(M.dot(A.dot(weights)), axis=0)),
	# print(np.sum(M.dot(A_x.dot(weights_x)), axis=0)),
	# print(np.sum(M.dot(A_y.dot(weights_y)), axis=0)),
	# print(np.sum(M.dot(A_z.dot(weights_z)), axis=0)),

	
	
	ind = l.index(max(l))

	if ind == 0:
		return orig_v, vertices
	elif ind == 1:
		return orig_v.dot(rx), vertices.dot(rx)
	elif ind == 2:
		return orig_v.dot(ry), vertices.dot(ry)
	elif ind == 3:
		return orig_v.dot(rz), vertices.dot(rz)

# Iterated residual fitting

def IRF(orig_v, vertices, faces, max_degree = 16):
	num_harm = max_degree**2
	sigma = 0.001

	theta = np.arccos(vertices[:,2])
	phi = np.arctan2(vertices[:,1],vertices[:,0])

	orig_v = normalize_area(orig_v, faces)

	# weighted least squares with mass matrix to account for unequal mesh resolution.
	W = igl.massmatrix(orig_v, faces, igl.MASSMATRIX_TYPE_BARYCENTRIC)

	


	weights = np.zeros( (num_harm,3) )
	i = 0
	A = []

	residual = np.copy(orig_v)

	for l in range(0, max_degree):
		Y = []
		for m in range(-l,l+1):
			y = sph_real(l, m, phi, theta)

			A.append(y)
			Y.append(y)
			

		smooth = 1#np.exp(-sigma*l*(l+1))

		Y = np.vstack(Y).T

		#w = np.linalg.solve(Y.T.dot(Y),Y.T.dot(residual))
		w = np.linalg.solve(Y.T.dot(W.dot(Y)),Y.T.dot(W.dot(residual)))

		residual = residual - Y.dot(w)

		i1 = l**2
		i2 = (l+1)**2
		weights[i1:i2] = smooth*w

	A = np.vstack(A).T

	#err = A.dot(weights) - orig_v
	#print("mean reconstruction error: " + str(np.mean(norm(err, axis=1))))

	return weights, A

def sum_weights(w):
	N = norm(w, axis=1)
	max_degree = int(np.sqrt(w.shape[0]))
	outnorms = np.zeros(max_degree)
	for l in range(0, max_degree):
		i1 = l**2
		i2 = (l+1)**2
		outnorms[l] = norm(N[i1:i2])

	return outnorms

def weights_canonical(w):
	outnorms = np.array([-w[3][0],-w[1][1],w[2][2]])

	outnorms = np.append(outnorms, w[4:].flatten())

	return outnorms

def weights_flatten(w):

	#print(np.vstack([-w[3],-w[1],w[2]]).T)
	outnorms = w.flatten()

	return outnorms


