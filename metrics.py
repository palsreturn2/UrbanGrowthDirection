import numpy as np
import math

def change_metric(R,Bt,Btnxt, Btpred):
	shp = Bt.shape
	A=0
	B=0
	C=0
	D=0
	E=0
	for i in range(0,shp[0]):
		for j in range(0,shp[1]):
			if(Bt[i][j]-Btnxt[i][j]!=0):
				if(Btpred[i][j]-Bt[i][j]!=0):
					B=B+1
				else:
					A=A+1
			if(Bt[i][j]-Btnxt[i][j]==0):
				if(Btpred[i][j]-Bt[i][j]!=0):
					D=D+1
				else:
					E=E+1
			if(Btpred[i][j]!=Btnxt[i][j]):
				C=C+1
	print A,B,C,D,E
	return float(B)/(A+B+C+D), float(B)/(A+B+C), float(B)/(B+C+D), float(B+E)/(A+B+C+D+E)

def angle_rmse(A,B):
	pi = 3.14
	D = A-B
	s = 0
	for i in range(0,D.shape[0]):
		if(abs(D[i])>pi):
			D[i] = 2*pi-abs(D[i])
			s = s+D[i]*D[i]
		else:
			s = s+D[i]*D[i]
	return math.sqrt(s/float(D.shape[0]))

