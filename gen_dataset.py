import numpy as np
import geninput as INPUT
from skimage import io
import sys

class Dataset:
	def __init__(self):
		self.X=[]
		self.Y=[]
		self.ndim=0
		self.nf=0
		self.normalized = True
		self.labelled = True
		
	def create_window(self,raster,posx,posy):		
		[row,col]=raster.shape
		
		xmin = posx-1
		ymin = posy-1
		xmax = posx+1
		ymax = posy+1
	
		window=np.zeros((5))
		if(ymin>=0):
			window[1] = raster[posx][ymin]
		if(xmin>=0):	
			window[2] = raster[xmin][posy]
		if(ymax<col):
			window[3] = raster[posx][ymax]
		if(xmax<row):
			window[4] = raster[xmax][posy]
		window[0] = raster[posx][posy]
		return window
		
	def create_dataset(self, raster, Bt, Btnxt, distancemaps):
		shp = raster.shape
		W = []
		D = []
		for i in range(0,shp[1]):
			for j in range(0,shp[2]):
				if(raster[0][i][j]!=0):
					if(distancemaps!=[]):
						x = np.zeros(len(distancemaps))
						for k in range(len(distancemaps)):
							x[k] = distancemaps[k][i][j]
						D.append(x)
					win = self.create_window(Bt,i,j)
					W.append(np.ndarray.flatten(win))					
					self.Y.append(Btnxt[i][j])
		D = np.array(D)
		W = np.array(W)
		if(distancemaps!=[]):
			D = D/(np.max(D,axis=0)-np.min(D,axis=0))
			self.X =np.concatenate([W,D,np.ones([D.shape[0],1])],axis=1)
		
		else:
			self.X = np.concatenate([W,np.ones([W.shape[0],1])],axis=1)
		return np.array(self.X),np.array(self.Y)
			

def ageBuiltUp(R, Bt, Btnxt, age):
	shp = Bt.shape
	Bt = Bt*age
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			if(R[0][i][j]!=0):
				if(Bt[i][j]==0):
					Bt[i][j]=-age
				if(Bt[i][j]>0 and Btnxt[i][j]>0):
					Btnxt[i][j] = Bt[i][j]+age
				if(Bt[i][j]>0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = 0
				if(Bt[i][j]<=0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = Bt[i][j]-age
				if(Bt[i][j]<=0 and Btnxt[i][j]>0):
					Btnxt[i][j] = age
	return Bt,Btnxt

if __name__=="__main__":
	D = Dataset()	
	
	data_folder = sys.argv[1]
	
	if(data_folder == ''):
		print 'Please enter the name of the data folder'
		exit()
	
	R = INPUT.give_raster(data_folder+'/1999.tif')

	D_F = INPUT.give_raster(data_folder+'/Distancemaps_F.tif')[0]

	D_B = INPUT.give_raster(data_folder+'/Distancemaps_B.tif')[0]

	D_W = INPUT.give_raster(data_folder+'/Distancemaps_W.tif')[0]

	#E = INPUT.give_raster(data_folder+'/Elevationmap.tif')[0]

	D_R = INPUT.give_raster(data_folder+'/Distancemaps_R.tif')[0]

	Bt = INPUT.give_raster(data_folder+'/cimg2000.tif')[0]

	Btnxt = INPUT.give_raster(data_folder+'/cimg2010.tif')[0]

	Bt = (Bt/255)
	Btnxt = (Btnxt/255)
	Bt,Btnxt = ageBuiltUp(R,Bt,Btnxt,0.1)

	'''io.imshow(np.transpose(Bt))
	io.show()'''

	dmaps = [D_F,D_B,D_W,D_R]

	X,Y = D.create_dataset(R,Bt,Btnxt,dmaps)

	np.save('X.npy',X)
	np.save('Y.npy',Y)

		
		
