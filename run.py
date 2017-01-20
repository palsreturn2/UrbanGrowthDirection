import gen_dataset as DATASET
from gen_dataset import Dataset
import train as TRAIN
import direction as DIRECTION
import sys
import geninput as INPUT
import naive as NAIVE

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
	
	#Training model
	params, model = TRAIN.train(X,Y)
	
	#Testing the model
	TRAIN.test(X, model, R, Bt, Btnxt, DV_G)
	
	#Generating direction vectors
	DIRECTION.visualize3(R,params)
