import pgdb
import numpy as np
from osgeo import gdal
import geninput as INPUT

def pixel2coord(ds,col,row):
	c, a, b, f, d, e = ds.GetGeoTransform()
	xp = a * col + b * row + a * 0.5 + b * 0.5 + c
	yp = d * col + e * row + d * 0.5 + e * 0.5 + f
	return xp,yp

def getDistanceMaps(R,raster,id_dn) :
	hostname = 'localhost'
	username = 'postgres'
	password = 'ubuntu'
	database = 'landuse'
	
	row = raster.RasterXSize
	col = raster.RasterYSize
	
	D = np.zeros((row,col))
	
	conn = pgdb.connect(host=hostname, user=username, password=password, database=database)
	cur = conn.cursor()
	print "Starting to generate distance map for id = %d"%id_dn
	cur.execute("create view class_landuse as select geom from landuse where dn=%d and ST_Area(geom)>1000000"%id_dn)
	for i in range(0,row):
		for j in range(0,col):
			lat,lon = pixel2coord(raster,j,i)
			if(R[0][i][j]!=0):
				cur.execute("SELECT MIN(ST_Distance(ST_SetSRID(ST_MakePoint(%f,%f),32643),geom)) FROM class_landuse;" %(lat,lon))
				for distance in cur.fetchall():
					D[i][j] = distance[0]
					break	
	conn.close()
	return D

raster = gdal.Open('mumbai1999.tif')

R = INPUT.give_raster('mumbai1999.tif')

D_wb = getDistanceMaps(R,raster,1)

np.save('distancemaps_wb.npy',D_wb)

D_f = getDistanceMaps(R,raster,2)

np.save('distancemaps_f',D_f)

D_b = getDistanceMaps(R,raster,3)

np.save('distancemaps_b',D_b)



