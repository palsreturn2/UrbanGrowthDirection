import gen_dataset as DATASET
from gen_dataset import Dataset
import train as TRAIN
import direction as DIRECTION
import sys
import geninput as INPUT
import naive as NAIVE
import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn.metrics
import scipy.stats

def direction_summary(R, Bt, Btnxt, dmaps, chunk_size = 50):
	shp = Bt.shape
	PX = []
	PY = []
	DX = []
	DY = []
	GT = []
	DA = []
	mean_angle=[]
	median_angle = []
	mode_angle = []
	n_mean_angle = []
	n_median_angle = []
	for i in range(0, shp[0], chunk_size):
		for j in range(0, shp[1], chunk_size):
			if(R[0][i][j]!=0):
				startx = i
				starty = j
				endx = min(shp[0],i+chunk_size)
				endy = min(shp[1],j+chunk_size)
				Rchunk = R[:, startx:endx, starty:endy]
				Btchunk = Bt[startx:endx, starty:endy]
				Btnxtchunk = Btnxt[startx:endx, starty:endy]
				dmapschunk = [] 
				for k in range(0,len(dmaps)):
					dmapschunk.append(dmaps[k][startx:endx, starty:endy])
				D = Dataset()
				X,Y = D.create_dataset(Rchunk,Btchunk,Btnxtchunk,dmapschunk)
				w, model = TRAIN.train(X,Y)
				P = model.predict(X)
				alpha = w[0]+w[1]+w[2]+w[3]+w[4] - 1
				beta = w[4]-w[2]
				gamma = w[3]-w[1]
				PX.append((startx+endx)/2)
				PY.append((starty+endy)/2)
				DX.append(beta/math.sqrt(beta*beta+gamma*gamma+1))
				DY.append(gamma/math.sqrt(beta*beta+gamma*gamma+1))
				#window = NAIVE.create_window(Btnxt,(startx+endx)/2,(starty+endy)/2)
				#GT.append(NAIVE.get_direction_angle(NAIVE.get_direction(window, Bt[(startx+endx)/2][(starty+endy)/2])))
				DA.append(math.atan(gamma/beta))
				GT = NAIVE.gen_direction_angles(Rchunk, Btchunk, Btnxtchunk)
				M,N = NAIVE.eval_naive_mean_median(Rchunk, Btchunk, P)
				n_mean_angle.append(M)
				n_median_angle.append(N)
				mean_angle.append(np.mean(GT))
				median_angle.append(np.median(GT))
				#mode_angle.append(scipy.stats.mode(GT))
				
	print sklearn.metrics.mean_squared_error(np.array(DA),np.array(mean_angle))
	print sklearn.metrics.mean_squared_error(np.array(DA),np.array(median_angle))
	print sklearn.metrics.mean_squared_error(np.array(n_mean_angle),np.array(mean_angle))
	print sklearn.metrics.mean_squared_error(np.array(n_median_angle),np.array(median_angle))
	plt.imshow(np.transpose(R[0]))
	Q = plt.quiver(np.array(PX),np.array(PY),np.array(DX),np.array(DY))
	plt.show()
	return			
				
	
if __name__=="__main__":
	data_folder = sys.argv[1]
	
	if(data_folder == ''):
		print 'Please enter the name of the data folder'
		exit()
	
	#Load Datasets
	R = INPUT.give_raster(data_folder+'/1999.tif')
	Bt = INPUT.give_raster(data_folder+'/cimg2000.tif')[0]
	Btnxt = INPUT.give_raster(data_folder+'/cimg2010.tif')[0]
	D_F = INPUT.give_raster(data_folder+'/Distancemaps_F.tif')[0]
	D_B = INPUT.give_raster(data_folder+'/Distancemaps_B.tif')[0]
	D_W = INPUT.give_raster(data_folder+'/Distancemaps_W.tif')[0]
	#E = INPUT.give_raster(data_folder+'/Elevationmap.tif')[0]
	D_R = INPUT.give_raster(data_folder+'/Distancemaps_R.tif')[0]
	DV_G = NAIVE.gen_direction_angles(R,Bt,Btnxt)
	
	#Create Dataset
	Bt = Bt/255
	Btnxt = Btnxt/255
	Bt,Btnxt = DATASET.ageBuiltUp(R,Bt,Btnxt,0.1)
	dmaps = [D_F,D_B,D_W,D_R]
	D = Dataset()
	X,Y = D.create_dataset(R,Bt,Btnxt,dmaps)
	
	#Direction Summary
	direction_summary(R, Bt, Btnxt, dmaps, chunk_size = 100)
	exit()
	
	#Training model
	params, model = TRAIN.train(X,Y)
	
	#Testing the model
	TRAIN.test(X, model, R, Bt, Btnxt, DV_G)
	
	#Generating direction vectors
	DIRECTION.visualize3(R,params)
