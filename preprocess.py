import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import statistics as st

def rgb2gray(rgb):
	# convert rgb image to gray image
	# input: ndarray of rgb image
	# output: ndarray of gray scale image

	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

def intensity(gray):
	# calculate the average intensity of the gray image
	# imput: ndarray of gray scale image
	# output: np.float64 of average intensity 

	row = np.size(gray, 0)
	col = np.size(gray, 1)

	return np.sum(gray) / (row * col)

def calculateStoke(imgs):
	# calculate a Stoke vector for the input image
	# input: aseries of RGB image
	# output: a Stoke vector with length 4

	intensities = []

	for img in imgs:
		cur_gray = rgb2gray(img)
		intensities.append(intensity(cur_gray))
	s1 = sum(intensities) / 2
	s2 = intensities[0] - intensities[2]
	s3 = intensities[1] - intensities[3]

	Stokes = [s1, s2, s3, 0] / s1

	return Stokes

def imgToStokes(imgs, k):
	# calculate the stokes vector of a series of images
	# input: a list of images, with polarization angle 0, 45, 90, 135; k * k patch to be calculated
	# output: (height / k) * (width / k) * 4 dimension stokes vector set (ndarray)

	height = len(imgs[0])
	width = len(imgs[0][0])

	stokesSet = np.zeros((int(height / k), int(width / k), 4))

	for i in range(0, height - 1, k):
		for j in range(0, width - 1, k):
			curImg = st.sliceArray(imgs, i, j, k)
			curStoke = calculateStoke(curImg)

			stokesSet[int(i / k), int(j / k)] = curStoke

	return stokesSet

'''
test = mpimg.imread("rgb45.jpg")
test_gray = rgb2gray(test)
plt.imshow(test_gray, cmap = plt.get_cmap('gray'))
plt.show()
print(intensity(test_gray))

imgs = [mpimg.imread("rgb45.jpg")] * 4
test = [[[254, 254, 254], [254, 254, 254]], [[255, 253, 254], [255, 253, 254]]]

print(type(imgToStokes(imgs, 2)[0, 0]))
'''
#import os
#print(os.path.pardir)

img2stoke = True

if(img2stoke):
	img0 = mpimg.imread("../data/p27 h206SW/polarization/IMG_0643_0.JPG")
	img1 = mpimg.imread("../data/p27 h206SW/polarization/IMG_0644_45.JPG")
	img2 = mpimg.imread("../data/p27 h206SW/polarization/IMG_0645_90.JPG")
	img3 = mpimg.imread("../data/p27 h206SW/polarization/IMG_0646_135.JPG")

	imgs = [img0, img1, img2, img3]

	np.save("p27 h206S.npy", imgToStokes(imgs, 50))