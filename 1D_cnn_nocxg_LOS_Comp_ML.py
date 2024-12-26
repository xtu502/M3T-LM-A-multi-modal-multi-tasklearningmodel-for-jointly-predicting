# -*- coding: utf-8 -*-
"""
Studied and Created on  2018/3/1

@author: batch William
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime
from keras import optimizers, callbacks, regularizers
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib
matplotlib.use('agg')   #去除Xmanager软件来处理X11

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import math

import keras
import tensorflow as tf
import os,sys
os.getcwd()
# os.chdir("D:/30_Project_dat/45_potato_dete")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

starttime = datetime.datetime.now()

def reg_metric(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    Evar = explained_variance_score(y_true, y_pred)
    print("MAE:",mae)
    print("MSE:",mse)
    print("RMSE:",rmse)
    print("R Square:",r2)
    print("Evar:",Evar)
    
# 自定义度量函数
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#-----------CXR data---------------------------------------------------------------
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob


## Verify
#df_tsfeat.to_csv('D:/22_论文写作/2. Length of stay prediction/03_算法预研/Testdat/df_tsfeat.csv', index=False)
df_tsfeat = pd.read_csv('/home/junde/06_Tensor-decompositions/CNN_compression_with_MIMIC_RL/data/df_tsfeat_VBMF.csv')


#all_xray_df = pd.read_csv('/home/junde/06_Tensor-decompositions/CNN_compression_with_MIMIC_RL/data/mimic-cxr-2.0.0-split.csv')
PAAP = pd.read_csv('/data/datasets/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv')
raw_dicom = pd.read_csv('/home/junde/02_LOS/Testdat/mimic-cxr-2.0.0-split.csv')
PA_dicom = pd.merge(raw_dicom,PAAP[['dicom_id','ViewPosition']],on=['dicom_id'],how="left")
all_xray_df = PA_dicom[PA_dicom['ViewPosition'].isin(['PA', 'AP'])]  

all_image_paths = {os.path.basename(x): x for x in 
                   #glob(os.path.join('/home/junde/06_Tensor-decompositions/CNN_compression_with_MIMIC_RL/data/', 'Image_dataset', '*.jpg'))}
                   glob(os.path.join('/data/datasets/', 'resized', '*.jpg'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])

PD_Img = all_xray_df['dicom_id'].astype('str') + '.jpg'

all_xray_df['path'] = PD_Img.map(all_image_paths.get)
all_xray_df['path'] = PD_Img.map(all_image_paths.get)

all_xray_df = all_xray_df.loc[all_xray_df['path'].notnull()]
all_xray_df.sample(3)
print(all_xray_df.shape)


chexpert_df = pd.read_csv('/home/junde/06_Tensor-decompositions/CNN_compression_with_MIMIC_RL/data/mimic-cxr-2.0.0-chexpert.csv',header=None) #
X_id = chexpert_df.iloc[1:,0:2]
X_id.columns=['subject_id','study_id']
X_id = X_id.reset_index(drop=True)

X_lbl = chexpert_df.iloc[1:,2:]
X_lbl = X_lbl.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
X_lbl = X_lbl.values.astype(float)

act_trnlbl = [np.argmax(ii) for ii in X_lbl]
# act_trnlbl = np.array(act_trnlbl)

act_trnlbl = pd.DataFrame(act_trnlbl)
X_clss = pd.concat([X_id,act_trnlbl],axis=1)
X_clss.rename(columns={0:'labels'}, inplace = True)


s1 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
all_labels = ['Atelectasis', 'Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']
map_tbl = pd.DataFrame({'labels':s1, 'classes':all_labels})


xray_lbl = pd.merge(all_xray_df,X_clss.drop(columns=['study_id']),on="subject_id",how="inner")
xray_lbl = pd.merge(xray_lbl,map_tbl,on="labels",how="left")
# df_clean = df_tsfeat.join(df_crx, on='SUBJECT_ID', lsuffix='_left', rsuffix='_right')
xray_lbl.drop_duplicates(subset=['study_id','dicom_id'],keep='first',inplace=True)

#------------CXR-LoS数据merge------------------------------------------------------------
xray_lbl=pd.merge(xray_lbl,PAAP[['dicom_id','StudyDate','StudyTime']],left_on='dicom_id',right_on='dicom_id',how="inner")
icu_csv=pd.read_csv('/home/junde/06_Tensor-decompositions/CNN_compression_with_MIMIC/data/ICUSTAYS.csv')
df_tsftim=pd.merge(df_tsfeat,icu_csv[['SUBJECT_ID','HADM_ID','INTIME','OUTTIME']],left_on=['SUBJECT_ID','HADM_ID'],right_on=['SUBJECT_ID','HADM_ID'],how="left")
df_tsftim.drop_duplicates(subset=['SUBJECT_ID','HADM_ID'],keep='first',inplace=True)

cxr_merged_los = pd.merge(xray_lbl,df_tsftim,left_on="subject_id",right_on="SUBJECT_ID",how="inner")
cxr_merged_los.drop_duplicates(subset=['subject_id','dicom_id'],keep='first',inplace=True)
# combine study date time
cxr_merged_los['StudyTime'] = cxr_merged_los['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
cxr_merged_los['StudyDateTime'] = pd.to_datetime(cxr_merged_los['StudyDate'].astype(str) + ' ' + cxr_merged_los['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
cxr_merged_los.intime=pd.to_datetime(cxr_merged_los.INTIME)
cxr_merged_los.outtime=pd.to_datetime(cxr_merged_los.OUTTIME)
end_time = cxr_merged_los.outtime
cxr_merged_los_during = cxr_merged_los.loc[(cxr_merged_los.StudyDateTime>=cxr_merged_los.intime)&((cxr_merged_los.StudyDateTime<=end_time))]
cxr_merged_los_during.drop_duplicates(subset=['subject_id','dicom_id'],keep='first',inplace=True)
cxr_merged_los_during.drop_duplicates(subset=['subject_id','LOS'],keep='first',inplace=True)
cxr_merged_los_during = cxr_merged_los_during.dropna(axis = 0, subset = ['LOS'] )
#-----------------------------------------------------------------------------------
xray_lbl=cxr_merged_los_during


'''
#xray_lbl=pd.merge(xray_lbl,df_tsfeat[['SUBJECT_ID','LOS', 'LOS_class']],left_on="subject_id",right_on="SUBJECT_ID",how="left")
xray_lbl=pd.merge(xray_lbl, df_tsfeat, left_on="subject_id",right_on="SUBJECT_ID",how="left")
xray_lbl.drop_duplicates(subset=['subject_id','dicom_id'],keep='first',inplace=True)
xray_lbl.drop_duplicates(subset=['subject_id','labels'],keep='first',inplace=True)
xray_lbl = xray_lbl.dropna(axis = 0, subset = ['labels'] )
# xray_lbl.sort_values(by=['subject_id','LOS'])
'''

trn_xray, tst_xray = xray_lbl[0:2500], xray_lbl[2501:]     # 1200, 10000,  2500
trn_xray.sort_values(by=['subject_id','labels'])
tst_xray.sort_values(by=['subject_id','labels'])



#-----------Pytorch 1D Table Modeling------------------------------------------------
trn_tbldat = trn_xray.drop(['ViewPosition','StudyDate','StudyTime','dicom_id', 'study_id', 'subject_id', 'split', 'path', 'labels',
        'classes', 'SUBJECT_ID','HADM_ID','Unnamed: 15', 'LOS', 'INTIME', 'OUTTIME', 'StudyDateTime'], axis=1) 
tst_tbldat = tst_xray.drop(['ViewPosition','StudyDate','StudyTime','dicom_id', 'study_id', 'subject_id', 'split', 'path', 'labels',
        'classes', 'SUBJECT_ID','HADM_ID','Unnamed: 15', 'LOS', 'INTIME', 'OUTTIME', 'StudyDateTime'], axis=1)
trn_tbldat.fillna(0, inplace=True)
tst_tbldat.fillna(0, inplace=True)

X_trn = trn_tbldat.values[:,1:53].astype(float)
Y_trn = trn_tbldat.values[:,0:1]
print('Y_trn=', Y_trn)

#encoder = LabelEncoder()
#Y_trn = encoder.fit_transform(Y_trn.ravel())

X_tst = tst_tbldat.values[:,1:53].astype(float)
Y_tst = tst_tbldat.values[:,0:1]
#encoder = LabelEncoder()
#Y_tst = encoder.fit_transform(Y_tst.ravel())

#----输入Net前，归一化输入数据-----------------------------------------
#scaler = MinMaxScaler(feature_range=(0, 1))
scaler = StandardScaler()
scaler.fit(X_trn)
X_trn = scaler.transform(X_trn)
X_tst = scaler.transform(X_tst)

#Y_scaler.fit(Y_trn.reshape(-1, 1))
#Y_trn = Y_scaler.transform(Y_trn.reshape(-1, 1))
#-------------------------------------------------------------------------




# Load libraries
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import lightgbm as ltb
# Test options and evaluation metric
seed = 7
scoring = 'accuracy' 


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
# rf = RandomForestRegressor(n_estimators=5)   # min_samples_split=40,  bootstrap=False
# rf = ltb.LGBMRegressor(n_estimators=5)       # , num_leaves=3, silent=True
# rf = XGBRegressor(n_estimators=70)
#rf = GradientBoostingRegressor(n_estimators=70)
#rf = SVR() 
# rf = DecisionTreeRegressor()        # max_depth=1
rf = MLPRegressor()   #max_iter=10 solver=’sgd’, activation=’relu’,alpha=1e-4,hidden_layer_sizes=(50,50), random_state=1,,learning_rate_init=.1)
#rf = KNeighborsRegressor(n_neighbors=6)
#rf = KNeighborsRegressor()
rf.fit(X_trn, Y_trn)

# y_test_pred = rf.predict(X_test)
#y_test_pred = cross_val_predict(rf, X_test, y_test, cv=3)




#------模型预测-------------------------
Y_trn_preds = rf.predict(X_trn)
#Y_trn_preds = cross_val_predict(rf, X_trn, Y_trn, cv=5)
print('-------------train res---------------')
reg_metric(Y_trn, Y_trn_preds)
plt.plot(Y_trn_preds)
plt.plot(Y_trn)
plt.show()


#-----测试集------------------
Y_tst_preds = rf.predict(X_tst)
#Y_tst_preds = cross_val_predict(rf, X_tst, Y_tst, cv=5)
print('-------------test res---------------')
reg_metric(Y_tst, Y_tst_preds)
plt.plot(Y_tst_preds)
plt.plot(Y_tst)
plt.show()

endtime = datetime.datetime.now()
print (endtime - starttime)
print ((endtime - starttime).seconds)  
