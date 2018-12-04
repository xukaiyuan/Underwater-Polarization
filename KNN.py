import heapq
import matplotlib.pyplot as plt
import numpy as np 
import statistics as st 
def distance(p1, p2):
	return st.angleDifference(p1, p2)
	'''
	p1 = (p1 + st.pi) % (2*st.pi)
	p2 = (p2 + st.pi) % (2*st.pi)
	return (p1-p2) % (2*st.pi)
	'''

def getaverage(pq, k=1000):
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



	for i in range(len(newpq)):
		ratio.append(1/newpq[i][0])
	sumnum = sum(ratio)
	ratio = np.array(ratio)/sumnum
	idx = 0
	jdx = 0
	for i in range(len(newpq)):
		idx += newpq[i][1]*ratio[i]
		jdx += newpq[i][2]*ratio[i]
	print(idx)
	print(jdx)



def getknn(base, k, datapoint):
	rowbase = len(base)
	colbase = len(base[0])
	pq = []
	for i in range(50):
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

def myplot(data, base):
	xarr = []
	yarr = []
	for i in range(len(data)):
		xarr.append(data[i][1])
		yarr.append(data[i][2])
	getaverage(data)

	plt.plot(xarr, yarr, 'ro')
	#plt.axis([0, len(base[0]), 0, len(base)])
	plt.show()

def knn(data, base):
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
	myplot(finaldata, base)


#mydata = np.random.rand(10, 20)
#mybase = np.random.rand(91,361)

#mydata = mydata.tolist()
#mybase = mybase.tolist()

#knn(mydata, mybase)
data_27_206 = np.load('p27 h190S_aop.npy')
data_27_206 = data_27_206.tolist()



base_27_206 = np.load('stokeTableAngle_190.npy')
#base_27_190 = np.load('test.npy')
base_27_206 = base_27_206.tolist()
plt.subplot(211)
plt.imshow(data_27_206)
plt.subplot(212)
plt.imshow(base_27_206)

plt.colorbar()
plt.show()

knn(data_27_206,base_27_206)

#plt.plot(newdata)
#plt.show()

