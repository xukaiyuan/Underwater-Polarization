'''
main function of the project of underwater navigation based on polarization 
'''

import matplotlib.image as mpimg
import numpy as np
from preprocess import *
from model import *
from KNN import *

# flags to control different session
img2stoke = True
image_aop = True
lookUpTable = True
lookUpTable_aop = True
knn = False
plot = False

# path of the images
path = "../data/p27 h239SW/polarization/"

# name of the raw Stoke vector file
nameOfStoke = "p27 h239SW.npy"

# name of angle of polarization of Image Stoke vector
nameOfAopImage = "p27 h239SW_aop.npy"

# name of look up table of Stoke vectors
nameOfLutStoke = "stokeVector_p27_239.npy"

# name of look up tabel of angle of polarization
nameOfAop = "stokeTableAngle_p27_239.npy"

'''
heading and pitch of the camera
heading is measured eastward with North as 0
pitch is measured from horizon with horizon as 0
'''
heading = deg2rad(239)
pitch = deg2rad(27)

if(img2stoke):
	img0 = mpimg.imread(path + "IMG_0657_0.JPG")
	img1 = mpimg.imread(path + "IMG_0658_45.JPG")
	img2 = mpimg.imread(path + "IMG_0659_90.JPG")
	img3 = mpimg.imread(path + "IMG_0660_135.JPG")

	imgs = [img0, img1, img2, img3]

	np.save(nameOfStoke, imgToStokes(imgs, 50))

# load and initialize the phtsical model
optical = model()
if(image_aop):
	stokeImage = np.load(nameOfStoke)
	angle = optical.stokeToAngleOfPolarization(stokeImage)
	np.save(nameOfAopImage, angle)

if(lookUpTable):
	stokeTable = np.zeros((91, 361, 4))

	for i in range(0, 91):
		for j in range(-180, 181):
			stokeTable[i, j] = optical.stokeVector(deg2rad(j), deg2rad(i), 1, 1.33, heading, pitch)

	np.save(nameOfLutStoke, stokeTable)

if(lookUpTable_aop):
	stokeVector = np.load(nameOfLutStoke)
	stokeAngle = optical.stokeToAngleOfPolarization(stokeVector)
	np.save(nameOfAop, stokeAngle)

if(knn):
	knn(nameOfAopImage, nameOfAop)

if(plot):
	plot_look_up(nameOfAopImage, nameOfAop)
