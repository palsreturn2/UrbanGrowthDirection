import sys
import geninput as INPUT
import numpy as np
from skimage import io

def create_window(raster,posx,posy):		
	[row,col]=raster.shape
		
	xmin = posx-1
	ymin = posy-1
	xmax = posx+1
	ymax = posy+1
	
	window=np.zeros((9))
	if(ymin>=0):
		window[1] = raster[posx][ymin]
		if(xmin>=0):
			window[2] = raster[xmin][ymin]
	if(xmin>=0):	
		window[3] = raster[xmin][posy]
		if(ymax<col):
			window[4] = raster[xmin][ymax]
	if(ymax<col):
		window[5] = raster[posx][ymax]
		if(xmax<row):
			window[6] = raster[xmax][ymax]
	if(xmax<row):
		window[7] = raster[xmax][posy]
		if(ymin>=0):
			window[8] = raster[xmax][ymin]
	window[0] = raster[posx][posy]
	return window

def get_direction(W,v):
	return np.argmax(np.abs(W[1:9]-v))+1

def get_direction_angle(d):
	pi = 3.14
	if(d==1):
		return -pi/2
	elif(d==2):
		return -3*pi/4
	elif(d==3):
		return pi
	elif(d==4):
		return 3*pi/4
	elif(d==5):
		return pi/2
	elif(d==6):
		return pi/4
	elif(d==7):
		return 0
	elif(d==8):
		return -pi/4
	else:
		return 0

def gen_direction_angles(R,Bt,Btnxt):
	shp = R.shape
	DV = []
	for i in range(shp[1]):
		for j in range(shp[2]):
			if(R[0][i][j]!=0):
				w = create_window(Btnxt,i,j)
				d = get_direction(w,Bt[i][j])
				DV.append(get_direction_angle(d))
	return np.array(DV)
	
def ageBuiltUp(R, Bt, Btnxt, age):
	shp = Bt.shape
	Bt = Bt*age
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			if(R[0][i][j]!=0):
				if(Bt[i][j]>0 and Btnxt[i][j]>0):
					Btnxt[i][j] = Bt[i][j]+age
				if(Bt[i][j]>0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = 0
				if(Bt[i][j]<=0 and Btnxt[i][j]<=0):
					Btnxt[i][j] = Bt[i][j]-age
				if(Bt[i][j]<=0 and Btnxt[i][j]>0):
					Btnxt[i][j] = age

if __name__ == "__main__":
	data_folder = sys.argv[1]
	if(data_folder == ''):
		print 'Please enter the name of the data folder'
		exit()
	R = INPUT.give_raster(data_folder+'/1999.tif')
	Bt = INPUT.give_raster(data_folder+'/cimg2000.tif')[0]
	Btnxt = INPUT.give_raster(data_folder+'/cimg2010.tif')[0]
	Bt = (Bt/255)
	Btnxt = (Btnxt/255)

	ageBuiltUp(R,Bt,Btnxt,0.01)

	io.imshow(np.transpose(Btnxt),cmap = 'rainbow_r')
	io.show()
	exit()
	DV = gen_direction_angles(R,Bt,Btnxt)

	np.save('DirectionAnglesGT.npy', DV)

		
	
