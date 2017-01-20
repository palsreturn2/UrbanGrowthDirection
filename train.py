import geninput as INPUT
import numpy as np
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from skimage import io
import skimage.filters
import naive
import time
import metrics
import sys
import math

def train(X,Y):
	model = SGDRegressor(epsilon=0.1, n_iter=50)
	print 'Training started'
	model.fit(X[0:int(0.3*X.shape[0])],Y[0:int(0.3*X.shape[0])])
	print 'Training ended'
	#print 'Regression accuracy (MSE) ',math.sqrt(sklearn.metrics.mean_squared_error(Y[int(0.7*X.shape[0]):X.shape[0]],P[int(0.7*X.shape[0]):X.shape[0]]))
	params = model.coef_
	return params, model

def test(X,model,R, Bt, Btnxt, DV_G):
	P = model.predict(X)
	Btpred = np.zeros((R.shape[1],R.shape[2]))
	k=0
	for i in range(Btnxt.shape[0]):
		for j in range(Btnxt.shape[1]):
			if(R[0][i][j]!=0):
				Btpred[i][j] = P[k]
				k=k+1
	print metrics.change_metric(R,Bt,Btnxt,Btpred)
	start_time = time.time()
	DV_P = np.ndarray.flatten(naive.gen_direction_angles(R,Bt,Btpred))
	print 'Time ',time.time()-start_time
	print sklearn.metrics.mean_squared_error(DV_G,DV_P)
	#io.imsave('./results/BtnxtRBWFE.png', np.transpose(Btnxt))
	#io.imshow(np.transpose(Btpred))
	#io.show()

if __name__ == "__main__":
	X = np.load('X.npy')
	Y = np.load('Y.npy')
	data_folder = sys.argv[1]
	
	if(data_folder == ''):
		print 'Please enter the name of the data folder'
		exit()
	Bt = INPUT.give_raster(data_folder+'/cimg2000.tif')[0]
	Btnxt = INPUT.give_raster(data_folder+'/cimg2010.tif')[0]
	Bt = Bt/255
	Btnxt = Btnxt/255
	

	model = SGDRegressor(epsilon=0.1, n_iter=50)
	#model = SVR(kernel = 'linear',C = 10)
	print 'Training started'
	model.fit(X[0:int(0.3*X.shape[0])],Y[0:int(0.3*X.shape[0])])
	#model.fit(X,Y)
	print 'Training ended'

	print model.coef_

	params = model.coef_

	P = model.predict(X)
	
	
	print 'Regression accuracy (MSE) ',sklearn.metrics.mean_squared_error(Y[int(0.7*X.shape[0]):X.shape[0]],P[int(0.7*X.shape[0]):X.shape[0]])

	R = INPUT.give_raster(data_folder+'/1999.tif')
	Btpred = np.zeros((R.shape[1],R.shape[2]))

	k=0
	for i in range(Btnxt.shape[0]):
		for j in range(Btnxt.shape[1]):
			if(R[0][i][j]!=0):
				Btpred[i][j] = P[k]
				k=k+1
	#Btpred[Btpred>0] = 1
	#Btpred[Btpred<0] = 0
	
	io.imshow(np.transpose(Btpred))
	io.show()
	print metrics.change_metric(R,Bt,Btnxt,Btpred)
	exit()
	
	start_time = time.time()
	DV_P = np.ndarray.flatten(naive.gen_direction_angles(R,Bt,Btpred))
	print 'Time ',time.time()-start_time
	DV_G = np.ndarray.flatten(np.load('DirectionAnglesGT.npy'))
	
	#print np.max(DV_P), np.max(DV_G)
	
	print sklearn.metrics.mean_squared_error(DV_G,DV_P)
	#io.imsave('./results/BtnxtRBWFE.png', np.transpose(Btnxt))
	io.imshow(np.transpose(Btpred))
	io.show()

