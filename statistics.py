import numpy as np
from pylab import *

def norm_cross(x, y):
	# calculate the nomalized cross product of a and b
	# input: 2 vectors with same dimension
	# output: normalized cross product

	crossed = cross(x, y)
	return crossed / norm(crossed)

def dotProduct(a, b, keepdims = True):
	# calculate the dot product of two vector (ndarrays)
	# input: two ndarrays
	# output: interger of the inner product

	return np.sum(a * b, axis = -1, keepdims = keepdims)

def vectorAngle(v1, v2):
	# calculate the angle between vector v1 and vector v2
	# input: vectors v1 and v2
	# output: angle in rad

	return arccos(sum(v1 * v2, -1)/sqrt(sum(v1 ** 2, -1) * sum(v2 ** 2, -1)))

def transferStoke(x1, y1, x2, y2):
	# transfer a stoke vector from (x1, y1) to (x2, y2)

	matrix = [[dot(x2, x1), dot(x2, y1)], [dot(y2, x1), dot(y2, y1)]]
	matrix = np.array(matrix, copy = False)
	A = np.array([[1, 0, 0, 1],[1, 0, 0, -1],[0, 1, 1, 0],[0, 1j, -1j, 0]])

	return 0.5 * np.real(A.dot(np.kron(matrix, matrix.conj())).dot(A.T.conj()))

def sliceArray(array, i, j, k):
	# slice every elements in the input array
	# input: array need to be sliced (2D array); start index i in row; start index j in row; step k
	# output: array with each element sliced within [i: i + k][j: j + k]

	newArray = []

	for element in array:
		current = element[i: i + k, j: j + k]
		newArray.append(current)

	return newArray

def angleDifference(a1, a2, period = pi):
	# calculater the angle difference
	# input: two angles in rad
	# output: the difference in rad

	return ((a1 - a2 + period / 2) % period) - period / 2

def stokeToAngle(stoke):
	# transfer a stoke vector to the angle of polarization
	# input: a stoke vector of length 4, ndarray
	# output: an angle in rad

	s1 = np.take(stoke, 1)
	s2 = np.take(stoke, 2)

	return 0.5 * np.arctan2(s2,s1)