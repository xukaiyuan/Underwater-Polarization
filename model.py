import statistics as st
import numpy as np
from pylab import *

class model:
	# all the parameters for refraction event

	# water surface normal vector
	def __init__(self):
		self.n = array([0,0,1])

	def incidenceLight(self, azimuthAngle, zenithAngle):
		# simulate the parameters of incidence light
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); water normal vector: array of length 3
		# output: wave vector ki (dimension 3); track vector xi (dimension 3)

		ki = -array([sin(azimuthAngle) * sin(zenithAngle), cos(azimuthAngle) * sin(zenithAngle), cos(zenithAngle)])
		xi = st.norm_cross(self.n, ki)

		return ki, xi

	def refractionLight(self, azimuthAngle, zenithAngle, ni, nt):
		# simulate the parameters of refracted light
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt
		# output: wave vector kt (dimension 3); track vector xt (dimension 3); Mueller matrix component ts and tp;

		ki, xi = self.incidenceLight(azimuthAngle, zenithAngle)
		ka = ki - self.n * st.dotProduct(ki, self.n)

		kt = ka - self.n * np.sqrt(st.dotProduct(ki, ki) * (nt / ni) ** 2 - st.dotProduct(ka, ka))
		xt = xi
		ts = 2 * st.dotProduct(ki, self.n, keepdims = False) / st.dotProduct(ki + kt, self.n, keepdims = False)
		tp = 2 * ni * nt * st.dotProduct(ki, self.n, keepdims = False) / st.dotProduct(ni ** 2 * kt + nt ** 2 * ki, self.n, keepdims = False)

		return kt, xt, ts, tp

	def muellerRefraction(self, azimuthAngle, zenithAngle, ni, nt):
		# calculate the Mueller matrix for refraction event
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt
		# output: 4 by 4 Mueller matrix for refraction event

		_, _, ts, tp = self.refractionLight(azimuthAngle, zenithAngle, ni, nt)

		element1 = ts ** 2 + tp ** 2
		element2 = ts ** 2 - tp ** 2
		element3 = 2 * ts * tp

		M_R = 0.5 * np.array([[element1, element2, 0, 0], [element2, element1, 0, 0], [0, 0, element3, 0], [0, 0, 0, element3]])

		return M_R

	def vectorForCamera(self, heading, kt, pitch = 0):
		# calculate the vector of camera from the parameters of angles
		# input: heading in rad; pitch in rad; transmission vector kt
		# output: wave vector kc; x direction xc; y direction yc

		kc = -array([sin(heading) * cos(pitch), cos(heading) * cos(pitch), sin(pitch)]) * norm(kt)
		xc = st.norm_cross(self.n, kc)
		yc = st.norm_cross(kc, xc)

		return kc, xc, yc

	def vectorForScattering(self, heading, kt, pitch = 0):
		# calculate the vector of scattering from refraction
		# input: heading in rad; pitch in rad; transmission vector kt
		# output: y direction ys; x direction related to refraction: xsr; x direction related to camera: xsc

		kc, _, _ = self.vectorForCamera(heading, kt, pitch)

		ys = st.norm_cross(kt, kc)
		xsr = st.norm_cross(ys, kt)
		xsc = st.norm_cross(ys, kc)

		return ys, xsr, xsc

	def refractionToScattering(self, azimuthAngle, zenithAngle, ni, nt, heading, pitch = 0):
		# calculate the transfer matrix from refraction plane to scattering plane
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt;
		#		 heading in rad; pitch in rad
		# output: a 4 by 4 matrix transfer M_R to M_S

		kt, xt, _, _ = self.refractionLight(azimuthAngle, zenithAngle, ni, nt)

		_, xsr, _ = self.vectorForScattering(heading, kt, pitch)

		yt = st.norm_cross(kt, xt)
		ysr = st.norm_cross(kt, xsr)

		M_R_S = st.transferStoke(xt, yt, xsr, ysr)

		return M_R_S

	def muellerScattering(self, azimuthAngle, zenithAngle, ni, nt, heading, pitch = 0):
		# calculate the Mueller matrix for scattering event
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt;
		#		 heading in rad; pitch in rad
		# output: 4 by 4 Mueller matrix for scattering event

		kt, _, _, _ = self.refractionLight(azimuthAngle, zenithAngle, ni, nt)

		kc, _, _ = self.vectorForScattering(heading, kt, pitch)

		theta = st.vectorAngle(kt, kc)
		c = np.cos(theta)

		b = self.fourierForand(theta)

		element1 = (c ** 2 - 1) / (c ** 2 + 1)
		element2 = 2 * c / (c ** 2 + 1)

		M_S = np.array([[1, element1, 0, 0], [element1, 1, 0, 0], [0, 0, element2, 0], [0, 0, 0, element2]])

		return M_S, b

	def fourierForand(self, theta, npart = 1.08, mu = 3.483):
		# calculate the Fourier Forand coefficient for scattering Mueller matrix
		# input: vector angle between kt and kc in rad; npart, the particle index for volume scattering function (real, > 1);
		# mu, the Junge slope for volume scattering function (3-5)
		# output: the Fourier Forand coefficient

		n = np.real(npart)
		d = 4 * np.sin(theta / 2) ** 2 / (3 * (n - 1) ** 2)
		d_180 = 4 * np.sin(np.pi / 2) / (3*(n - 1) ** 2)
		v = (3 - mu) / 2
		dv = d ** v
		d_180v = d_180 ** v
		d1 = 1 - d
		dv1 = 1 - dv

		a = 1 / (4 * np.pi * dv * d1 ** 2)
		b = v * d1 - dv1 + (d * dv1 - v * d1) * np.sin(theta / 2) ** (-2)
		c = (1 - d_180v) * (3 * np.cos(theta) ** 2 - 1) / (16 * np.pi * (d_180 - 1) * d_180v)

		return a * b + c

	def scatteringToDetect(self, azimuthAngle, zenithAngle, ni, nt, heading, pitch = 0):
		# calculate the transfer matrix from scattering plane to detect plane
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt;
		#		 heading in rad; pitch in rad
		# output: a 4 by 4 matrix transfers M_S to M_D

		kt, _, _, _ = self.refractionLight(azimuthAngle, zenithAngle, ni, nt)

		_, xc, yc = self.vectorForCamera(heading, kt, pitch)
		ys, _, xsc = self.vectorForScattering(heading, kt, pitch)

		M_S_D = st.transferStoke(xsc, ys, xc, yc)

		return M_S_D

	def stokeVector(self, azimuthAngle, zenithAngle, ni, nt, heading, pitch = 0):
		# calculate detected Stoke vector
		# input: azimuth angle in rad (eastward from north); zenith angle in rad (down from vertical); index of air: ni; index of watter: nt;
		#		 heading in rad; pitch in rad
		# output: a stoke vector for length 4

		M_R = self.muellerRefraction(azimuthAngle, zenithAngle, ni, nt)
		M_R_S = self.refractionToScattering(azimuthAngle, zenithAngle, ni, nt, heading)

		M_S, b = self.muellerScattering(azimuthAngle, zenithAngle, ni, nt, heading)
		M_S_D = self.scatteringToDetect(azimuthAngle, zenithAngle, ni, nt, heading)

		stokeDetected = M_S_D.dot(b * M_S.dot(M_R_S.dot(M_R.dot([1,0,0,0]))))

		return stokeDetected

	def stokeToAngleOfPolarization(self, stokeDetected):
		# transfer detected Stoke vectors to angle of polarization
		# input: m * n * 4 Stoke vectors
		# output: m * n angle of polarization, all in rad

		row = stokeDetected.shape[0]
		col = stokeDetected.shape[1]

		angleOfPolarization = np.zeros((row, col))

		for i in range(row):
			for j in range(col):
				stoke = stokeDetected[i, j]
				angle = st.stokeToAngle(stoke)
				angleOfPolarization[i, j] = angle

		return angleOfPolarization