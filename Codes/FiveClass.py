###From Scientific Data 7(1):154
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy as sp
import ast
import wfdb
def load_raw_data(df, sampling_rate=100):
	data=[wfdb.rdsamp(f) for f in df.filename_lr]
	data=np.array([signal for signal,meta in data])
	return(data)


df=pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))
X=load_raw_data(df,100)
agg_df=pd.read_csv('scp_statements.csv',index_col=0)
agg_df=agg_df[agg_df.diagnostic==1]
def aggregate_diagnostic(y_dic):
	temp=[]
	for key in y_dic.keys():
		if key in agg_df.index:
			temp.append(agg_df.loc[key].diagnostic_class)
		if len(temp)>0:
			return temp[0]
		else:
			return('NA')
df['diagnostic_superclass']=df.scp_codes.apply(aggregate_diagnostic)
print(df.head())

#test_fold=10
#train_fold=[1,2,3]
#X_train=X[np.where(df.strat_fold.isin(train_fold))]
#X_train_list=df[df.strat_fold.isin(train_fold)].filename_lr
#y_train=df[df.strat_fold.isin(train_fold)].diagnostic_superclass
#dl=list(df[df.strat_fold.isin(train_fold)].diagnostic_superclass)

#train_csv=df.loc[df['strat_fold'].isin(train_fold)]
#train_csv = train_csv[['filename_lr', 'diagnostic_superclass']]
classd= defaultdict(lambda: np.nan, {'NORM':0,'STTC':1,'MI':2,'CD':3,'HYP':4})
dfn=df[['diagnostic_superclass']].copy()

dfn['superclass_num']=dfn['diagnostic_superclass'].map(classd)
dfn.to_csv('5class.csv')


