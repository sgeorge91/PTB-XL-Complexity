import wfdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import frappy as fp
import scipy.stats as sts
import neurokit2 as nk
import pyunicorn.timeseries as put
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

def find_tau(x):
	mu=np.mean(x)
	sig2=np.var(x)
	xn=x-mu
	acf=np.correlate(xn,xn,'full')[len(xn)-1:]
	acf=acf/sig2/len(xn)
	tau=np.where(acf<(1./np.exp(1)))[0][0]
	return(tau)
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
		tau=find_tau(df_detII[0])
		entropy = nk.complexity_mse(df_detII[0], dimension=3, delay=tau)
		crpen, info = nk.entropy_permutation(df_detII[0], delay=tau, dimension=3, conditional=True, algorithm=nk.entropy_renyi, alpha=2)
		lzv,info=nk.complexity_lempelziv(df_detII[0], delay=tau, dimension=3)
		q=[ddict[n],entropy[0],crpen,lzv]
		a.append(q)
	except Exception as error:
		print('error is',error)
dff= pd.DataFrame(a, columns=['id','mse','PermEnt','LZV'])
dff.to_csv('mse_ptb_v1.dat', index=False)


