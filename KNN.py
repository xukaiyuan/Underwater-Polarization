import heapq
import matplotlib.pyplot as plt
import numpy as np 
import sys
import statistics as st 

def distance(p1, p2):
	# return the angle difference of two polarization angles
	# input: angle of polarizations in rad
	# output: difference of angle in rad, from 0 to pi
	return abs(st.angleDifference(p1, p2))

def getaverage(pq, k=1000):
	# get the average of 1000 closest points and then plot
	# input: pq is a priority queue
	# no output but the plot of the points 
	ratio = []
	newpq = []
	for i in range(len(pq)):
		if i < k:
			heapq.heappush(newpq, pq[i])
		else:
			tmp = heapq.heappop(newpq)
			if pq[i][0] < tmp[0]:
				heapq.heappush(newpq, tmp)
			else:
				heapq.heappush(newpq, pq[i])

	xarr = []
	yarr = []
	for i in range(len(newpq)):
		if newpq[i][0] != 0:
			xarr.append(newpq[i][2]-180.0)
			yarr.append(newpq[i][1])
			ratio.append(1/newpq[i][0])

	# calculate weighted mean
	sumnum = sum(ratio)
	ratio = np.array(ratio)/sumnum
	idx = 0
	jdx = 0
	for i in range(len(ratio)):
		idx += newpq[i][1]*ratio[i]
		jdx += newpq[i][2]*ratio[i]
	print('predicted solar zenith angle %f' %idx)
	print('predicted heading relative to the Sun' %(jdx-180))

	plt.plot(xarr, yarr, 'ro')
	plt.xlabel('Heading relative to Sun')
	plt.ylabel('Solar Zenith Angle')
	plt.show()

def getknn(base, k, datapoint):
	# every data point, get 3 closest point from lookup table
	# input: look-up table, number of points need to choose in one comparison, the data
	# output: a priority queue of the results
	rowbase = len(base)
	colbase = len(base[0])
	pq = []
	for i in range(rowbase):
		for j in range(colbase):
			if i == 0:
				continue
			curdist = distance(datapoint, base[i][j])

			if len(pq) < k:
				heapq.heappush(pq, (-curdist, i, j))
			else:
				tmp = heapq.heappop(pq)
				curmin = -1.0*tmp[0]
				if curdist < curmin:
					heapq.heappush(pq, (-curdist, i, j))
				else:
					heapq.heappush(pq, tmp)
	return pq 
	
def knn(npdata, npbase):
	# knn algorithm to find the cloestest points in the comparison
	# input: angle of polarization of images (ndarray), look-up table of angle of polarization (ndarray)
	# no output but the plots and the average results
	data = npdata.tolist()
	base = npbase.tolist()
	rowdata = len(data)
	coldata = len(data[0])
	k = 3
	finaldata = []
	for i in range(rowdata):
		for j in range(coldata):
			kpoints = getknn(base, k, data[i][j])
			finaldata = finaldata + kpoints
		print('finish row: %d' %i)

	print('begin printing')
	getaverage(finaldata)

def plot_look_up(data, base)
	# plot data and lookup table
	# input: angle of polarization of images (ndarray), look-up table of angle of polarization (ndarray)
	# no output but the plot of heatmap of the look-up table
	data_lst = data.tolist()
	base_lst = base.tolist()
	plt.subplot(211)
	plt.imshow(data_lst)
	plt.title('Polarized angle from photo')
	plt.subplot(212)
	plt.imshow(base_lst, extent=[-180.0,180.0,90.0,0.0])
	plt.title('Lookup table for AoP of Sun')
	plt.xlabel('Heading relative to Sun')
	plt.ylabel('Solar Zenith Angle')
	plt.colorbar()
	plt.show()

