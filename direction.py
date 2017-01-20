import numpy as np
import geninput as INPUT 
import gen_dataset as DATASET
import sklearn.metrics
import scipy.misc
import matplotlib.pyplot as plt
import math
import time
import sys

def visualize(Bt):
	shp = Bt.shape
	V = np.zeros(shp)
	DM = np.zeros(((shp[0]/32)+1,(shp[1]/32)+1))
	w=np.array([-2.15078193, 38.85633716, 41.98344035, 36.34808488, 39.66040379, 91.86686589, -169.45102728])
	
	alpha = w[0]+w[1]+w[2]+w[3]+w[4] - 1
	beta = w[4]-w[2]
	gamma = w[3]-w[1]
	
	objD = Dataset()
	
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			window = objD.create_window(Bt,i,j)
			B = window[0]
			gradxB = (window[4]-window[2])/2.0
			gradyB = (window[3]-window[2])/2.0
			par = [B*alpha, gradxB*beta, gradyB*gamma,-gradxB*beta, -gradyB*gamma]
			V[i][j] = np.argmax(par)
	
	for i in range(0,shp[0],32):
		for j in range(0,shp[1],32):
			a = np.ndarray.astype(np.ndarray.flatten(V[i:i+min(i+32,shp[0]), j:j+min(j+32,shp[1])]), dtype = np.int16)
			a = a[a!=0]
			if(a.shape[0]!=0):
				DM[i/32][j/32] = stats.mode(a)[0][0]
			
	return V,DM


def visualize2(R, X, w):
	shp = R.shape
	DV = np.zeros([2,shp[0],shp[1]])
	DA = []
	
	
	alpha = w[0]+w[1]+w[2]+w[3]+w[4] - 1
	beta = w[4]-w[2]
	gamma = w[3]-w[1]
	
	#xran = max(int(math.ceil(math.fabs(beta))),64)
	#yran = max(int(math.ceil(math.fabs(gamma))),64)
	
	k=0
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):			
			if(R[i][j]!=0):
				if(w.shape[0]>6):
					DV[0][i][j] = -beta/(alpha*X[k][0] + np.dot(w[5:w.shape[0]],X[k][5:w.shape[0]]))
					DV[1][i][j] = -gamma/(alpha*X[k][0] + np.dot(w[5:w.shape[0]],X[k][5:w.shape[0]]))
					DA.append(3.14 + math.atan(DV[0][i][j]/DV[1][i][j]))
				else:
					DV[0][i][j] = -beta/(alpha*X[k][0] + X[k][5]*w[5])
					DV[1][i][j] = -gamma/(alpha*X[k][0] + X[k][5]*w[5])
					DA.append(3.14 + math.atan(DV[0][i][j]/DV[1][i][j]))
				k=k+1
	return DV, DA

def visualize3(R,w):
	shp = R.shape
	alpha = w[0]+w[1]+w[2]+w[3]+w[4] - 1
	beta = w[4]-w[2]
	gamma = w[3]-w[1]
	plt.imshow(np.transpose(R[0]))
	Q = plt.quiver(np.array(shp[1]/2),np.array(shp[2]/2),np.array(-5*beta),np.array(-5*gamma))
	#qk = plt.quiverkey(Q, 0.5, 0.92, 2, '', labelpos='W', fontproperties={'weight': 'bold'})
	plt.show()	

if __name__ == "__main__":
	data_folder = sys.argv[1] 
	if(data_folder == ''):
		print 'Please enter the name of the data folder'
		exit()
	R = INPUT.give_raster(data_folder+'/1999.tif')
	X = np.load('X.npy')

	#w = [10.68581125, 31.3394232, 32.08895252, 30.51013695, 31.75016575, 227.43837872, -411.01570062]
	w=np.array([ 0.89909428, 0.15850894, 0.16689449, 0.15305532, 0.19495732, 0.0867942, -0.23138549, 0.09819625, -0.04861847, 0.00415151])

	start_time = time.time()
	Dvec, Dangle = visualize2(R[0],X,w)
	print 'Time ',time.time()-start_time

	#np.save('DistanceVector.npy', Dvec)
	#np.save('DirectionMap.npy', Dmap)

	#DA_GT = np.load('DirectionAnglesGT.npy')
	#print sklearn.metrics.mean_squared_error(Dangle,DA_GT)
	#Dvec = np.load('DistanceVector.npy')
	#Dmap = np.load('DirectionMap.npy')

	gradx = np.transpose(Dvec[0])
	grady = np.transpose(Dvec[1])


	U = []
	V = []
	X = []
	Y = []

	for i in range(0,gradx.shape[0],128):
		for j in range(0,gradx.shape[1],128):
			U.append(gradx[i][j])
			V.append(grady[i][j])
			X.append(i)
			Y.append(j)

	plt.imshow(np.transpose(R[0]))

	Q = plt.quiver(np.array(Y),np.array(X),np.array(U),np.array(V))
	qk = plt.quiverkey(Q, 0.5, 0.92, 2, '', labelpos='W', fontproperties={'weight': 'bold'})

	plt.show()	
	#plt.savefig('./results/UGDV2.png')	
	







