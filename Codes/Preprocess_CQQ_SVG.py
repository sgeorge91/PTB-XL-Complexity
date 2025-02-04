import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import scipy.stats as sts

def detrend(df):
	record_df_wo_trend = []
	adjusted_df = pd.DataFrame(index=df.index)                
	x = np.arange(len(df))  # Create an array of indices [0, 1, 2, ..., N-1]
	y = df.values           # Extract the values from the series
	# Fit a 20th-degree polynomial to the data to model the trend
	z = np.polyfit(x, y, 20)
	p = np.poly1d(z)       # Create a polynomial function from the coefficients
	trend = p(x)            # Evaluate the polynomial at each index to get the trend line
	y_adjusted = y - trend  # Subtract the trend from the original data to remove the baseline
	adjusted_df = y_adjusted  # Update the adjusted DataFrame with the baseline-corrected data
	record_df_wo_trend.append(adjusted_df)  # Add the adjusted data to the list
	return(record_df_wo_trend)

def MI(X,Y,Z):
	m1=nk.mutual_information(X[0],Y[0],bins=128,method="nolitsa")
	m2=nk.mutual_information(X[0],Z[0],bins=128,method="nolitsa")
	m3=nk.mutual_information(Y[0],Z[0],bins=128,method="nolitsa")
	return(m1,m2,m3)
def SC(X,Y,Z):
	rho1, p = sts.spearmanr(X[0], Y[0])
	rho2, p = sts.spearmanr(X[0], Z[0])
	rho3, p = sts.spearmanr(Y[0], Z[0])
	return(rho1,rho2,rho3)
	
a=[]
# read flash.dat to a list of lists
#datContent = [i.strip().split() for i in open("./00001_lr.dat").readlines()]
dfptb=pd.read_csv('ptbxl_database.csv')
df2=dfptb[['filename_lr','ecg_id']]
ddict = dict(zip(df2.filename_lr, df2.ecg_id))
fl=list(df2.filename_lr)
for n in fl:
	try:
		record = wfdb.rdrecord(n)
		#print(record)
		dfa=pd.DataFrame(record.p_signal, columns=record.sig_name)
		df_detII=detrend(dfa.II)
		df_detAVL=detrend(dfa.AVL)
		df_detV2=detrend(dfa.V2)
		m1,m2,m3=MI(df_detII,df_detAVL,df_detV2)
		r1,r2,r3=SC(df_detII,df_detAVL,df_detV2)
		q=[ddict[n],m1,m2,m3,r1,r2,r3]
		a.append(q)
	except:
		continue
dff= pd.DataFrame(a, columns=['id','MI-II-AVL','MI-II-V2','MI-V2-AVL','rho-II-AVL','rho-II-V2','rho-V2-AVL'])
dff.to_csv('CrossTimeSeries.dat', index=False)

