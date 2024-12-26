# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 21:59:00 2023

@author: 47874
"""

from keras.optimizers import SGD
from keras import optimizers, callbacks, regularizers
import datetime
from keras import optimizers
from sklearn.metrics import accuracy_score,recall_score,f1_score,classification_report
from keras.utils import np_utils 
import pandas as pd
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from keras import backend as K
import matplotlib
matplotlib.use('agg')   #去除Xmanager软件来处理X11

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import math
import random
from sklearn.neighbors import NearestNeighbors

import keras
import tensorflow as tf
import os,sys
os.getcwd()
# os.chdir("D:/30_Project_dat/45_potato_dete")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 1"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

starttime = datetime.datetime.now()


# Focal Loss函数
import tensorflow as tf        #定义多分类Focal函数
def focal_loss(gamma=2.):            #多分类，不带alpha
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed


#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))
 


    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
#        # train_acc        
        plt.plot(iters, self.accuracy[loss_type], 'coral', label='train acc', linewidth=2)     
        # val_acc
        plt.plot(iters, self.val_acc[loss_type], 'dodgerblue', label='val acc', linewidth=2) 

##        y = range(0,1.2,0.2)
#        y = np.arange(0.6,1,0.1)
        y = np.arange(0.8,1,0.05)
        plt.yticks(y) 
           
        plt.grid(axis="y")
        plt.xlabel(loss_type)
        plt.ylabel('accuracy')
        plt.legend(loc="best")    
        
        plt.savefig('fig_acc.png')            
#    if loss_type == 'epoch':
        plt.figure()       
        # val_acc
        plt.plot(iters, self.losses[loss_type], 'coral', label='train loss', linewidth=2)
        # val_loss
        plt.plot(iters, self.val_loss[loss_type], 'dodgerblue', label='val loss', linewidth=2)
        plt.grid(axis="y")
        plt.xlabel(loss_type)
        plt.ylabel('loss')
        plt.legend(loc="best")
        
#        y2 = np.arange(0,1.2,0.2)
        y2 = np.arange(0,0.2,0.05)
        plt.yticks(y2) 

        plt.savefig('fig_loss.png')        
        # plt.show()


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
    
# Focal Loss函数
import tensorflow as tf        #定义多分类Focal函数
def focal_loss(gamma=2.):            #多分类，不带alpha
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed

# 自定义度量函数
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

'''
def mimic_los_cleanup(adm_csv='ADMISSIONS.csv', patients_csv='PATIENTS.csv',
                      diagcode_csv='DIAGNOSES_ICD.csv', icu_csv='ICUSTAYS.csv',
                      verbose=True):
    
    # Import CSV tables
    df = pd.read_csv(adm_csv)
    df_pat = pd.read_csv(patients_csv)
    df_diagcode = pd.read_csv(diagcode_csv)
    df_icu = pd.read_csv(icu_csv)
    if verbose: 
        print('(1/5) Completed .csv imports')
        
    # Feature Engineering for Length of Stay (LOS) target variable
    # Convert admission and discharge times to datatime type
    df['ADMITTIME'] = pd.to_datetime(df['ADMITTIME'])
    df['DISCHTIME'] = pd.to_datetime(df['DISCHTIME'])
    # Convert timedelta type into float 'days', 86400 seconds in a day
    df['LOS'] = (df['DISCHTIME'] - df['ADMITTIME']).dt.total_seconds()/86400
    
    # Drop columns that are not needed for next steps  'ROW_ID', 'HAS_CHARTEVENTS_DATA', 'HOSPITAL_EXPIRE_FLAG'
    df.drop(columns=['DISCHTIME', 
                    'EDREGTIME', 'EDOUTTIME'
                    ], inplace=True)
    
    # Track patients who died at the hospital by admission event
    df['DECEASED'] = df['DEATHTIME'].notnull().map({True:1, False:0})
    
    # Hospital LOS metrics
    actual_mean_los = df['LOS'].loc[df['DECEASED'] == 0].mean() 
    actual_median_los = df['LOS'].loc[df['DECEASED'] == 0].median() 
    
    # Compress the number of ethnicity categories
    df['ETHNICITY'].replace(regex=r'^ASIAN\D*', value='ASIAN', inplace=True)
    df['ETHNICITY'].replace(regex=r'^WHITE\D*', value='WHITE', inplace=True)
    df['ETHNICITY'].replace(regex=r'^HISPANIC\D*', value='HISPANIC/LATINO', inplace=True)
    df['ETHNICITY'].replace(regex=r'^BLACK\D*', value='BLACK/AFRICAN AMERICAN', inplace=True)
    df['ETHNICITY'].replace(['UNABLE TO OBTAIN', 'OTHER', 'PATIENT DECLINED TO ANSWER', 
                             'UNKNOWN/NOT SPECIFIED'], value='OTHER/UNKNOWN', inplace=True)
    df['ETHNICITY'].loc[~df['ETHNICITY'].isin(df['ETHNICITY'].value_counts().nlargest(5).index.tolist())] = 'OTHER/UNKNOWN'

    # # Reduce categories to terms of religious or not
    # df['RELIGION'].loc[~df['RELIGION'].isin(['NOT SPECIFIED', 'UNOBTAINABLE'])] = 'RELIGIOUS'

    # Re-categorize NaNs into 'Unknown'
    df['MARITAL_STATUS'] = df['MARITAL_STATUS'].fillna('UNKNOWN (DEFAULT)')
    
    if verbose: 
        print('(2/5) Completed ADMISSIONS.csv cleanup and feature engineering.')
        
    # Feature Engineering for ICD9 code categories
    # Filter out E and V codes since processing will be done on the numeric first 3 values
    df_diagcode['recode'] = df_diagcode['ICD_CODE']
    df_diagcode['recode'] = df_diagcode['recode'][~df_diagcode['recode'].str.contains("[a-zA-Z]").fillna(False)]
    df_diagcode['recode'].fillna(value='999', inplace=True)
    df_diagcode['recode'] = df_diagcode['recode'].str.slice(start=0, stop=3, step=1)
    df_diagcode['recode'] = df_diagcode['recode'].astype(int)
    
    # ICD-9 Main Category ranges
    icd9_ranges = [(1, 140), (140, 240), (240, 280), (280, 290), (290, 320), (320, 390), 
                   (390, 460), (460, 520), (520, 580), (580, 630), (630, 680), (680, 710),
                   (710, 740), (740, 760), (760, 780), (780, 800), (800, 1000), (1000, 2000)]

    # Associated category names
    diag_dict = {0: 'infectious', 1: 'neoplasms', 2: 'endocrine', 3: 'blood',
                 4: 'mental', 5: 'nervous', 6: 'circulatory', 7: 'respiratory',
                 8: 'digestive', 9: 'genitourinary', 10: 'pregnancy', 11: 'skin', 
                 12: 'muscular', 13: 'congenital', 14: 'prenatal', 15: 'misc',
                 16: 'injury', 17: 'misc'}

    # Re-code in terms of integer
    for num, cat_range in enumerate(icd9_ranges):
        df_diagcode['recode'] = np.where(df_diagcode['recode'].between(cat_range[0],cat_range[1]), 
                num, df_diagcode['recode'])

    # Convert integer to category name using diag_dict
    df_diagcode['recode'] = df_diagcode['recode']
    df_diagcode['cat'] = df_diagcode['recode'].replace(diag_dict)
    
    # Create list of diagnoses for each admission
    hadm_list = df_diagcode.groupby('HADM_ID')['cat'].apply(list).reset_index()
    
    # Convert diagnoses list into hospital admission-item matrix
    hadm_item = pd.get_dummies(hadm_list['cat'].apply(pd.Series).stack()).sum(level=0)
    
    # Join back with HADM_ID, will merge with main admissions DF later
    hadm_item = hadm_item.join(hadm_list['HADM_ID'], how="outer")

    # Merge with main admissions df
    df = df.merge(hadm_item, how='inner', on='HADM_ID')
    
    if verbose: 
        print('(3/5) Completed DIAGNOSES_ICD.csv cleanup and feature engineering.')
    
    # Feature Engineering for Age and Gender
    # Convert to datetime type
    df_pat['DOD'] = pd.to_datetime(df_pat['DOD'])
    df_pat = df_pat[['SUBJECT_ID', 'DOD', 'GENDER']]
    df = df.merge(df_pat, how='inner', on='SUBJECT_ID')
    
    # Find the first admission time for each patient
    df_age_min = df[['SUBJECT_ID', 'ADMITTIME']].groupby('SUBJECT_ID').min().reset_index()
    df_age_min.columns = ['SUBJECT_ID', 'ADMIT_MIN']
    df = df.merge(df_age_min, how='outer', on='SUBJECT_ID')
    
    # Age is decode by finding the difference in admission date and date of birth
    # df['age'] = (df['ADMIT_MIN'] - df['DOD']).dt.days // 365
    df['age']  = ((df['ADMIT_MIN'].values  - df['DOD'].values).astype(np.int)/8.64e13//365).astype(np.int)
    df['age'] = np.where(df['age'] < 0, 90, df['age'])
    
    # Create age categories
    age_ranges = [(0, 13), (13, 36), (36, 56), (56, 100)]
    for num, cat_range in enumerate(age_ranges):
        df['age'] = np.where(df['age'].between(cat_range[0],cat_range[1]), 
                num, df['age'])
    age_dict = {0: 'newborn', 1: 'young_adult', 2: 'middle_adult', 3: 'senior'}
    df['age'] = df['age'].replace(age_dict)
    
    # Re-map Gender to boolean type
    df['GENDER'].replace({'M': 0, 'F':1}, inplace=True)
    
    if verbose: 
        print('(4/5) Completed PATIENT.csv cleanup and feature engineering.')
    
    # Feature engineering for Intensive Care Unit (ICU) category
    # Reduce ICU categories to just ICU or NICU
    df_icu['FIRST_CAREUNIT'].replace({'CCU': 'ICU', 'CSRU': 'ICU', 'MICU': 'ICU',
                                  'SICU': 'ICU', 'TSICU': 'ICU'}, inplace=True)
    df_icu['cat'] = df_icu['FIRST_CAREUNIT']
    icu_list = df_icu.groupby('HADM_ID')['cat'].apply(list).reset_index()
    icu_item = pd.get_dummies(icu_list['cat'].apply(pd.Series).stack()).sum(level=0)
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list['HADM_ID'], how="outer")
    df = df.merge(icu_item, how='outer', on='HADM_ID')
    
    # Cleanup NaNs
    # df['ICU'].fillna(value=0, inplace=True)
    # df['NICU'].fillna(value=0, inplace=True)
    df['Cardiac Vascular Intensive Care Unit (CVICU)'].fillna(value=0, inplace=True)
    df['Coronary Care Unit (CCU)'].fillna(value=0, inplace=True)
    df['Medical Intensive Care Unit (MICU)'].fillna(value=0, inplace=True)
    df['Medical/Surgical Intensive Care Unit (MICU/SICU)'].fillna(value=0, inplace=True)
    df['Neuro Intermediate'].fillna(value=0, inplace=True)
    df['Neuro Stepdown'].fillna(value=0, inplace=True)
    df['Neuro Surgical Intensive Care Unit (Neuro SICU)'].fillna(value=0, inplace=True)
    df['Surgical Intensive Care Unit (SICU)'].fillna(value=0, inplace=True)
    df['Trauma SICU (TSICU)'].fillna(value=0, inplace=True)
    
    if verbose: 
        print('(5/5) Completed ICUSTAYS.csv cleanup and feature engineering.')
        
    ## Remove deceased persons as they will skew LOS result
    #df = df[df['DECEASED'] == 0]

    # Remove LOS with negative number, likely entry form error
    df = df[df['LOS'] > 0]    
    
    # Drop unused columns, e.g. not used to predict LOS           'SUBJECT_ID',  'DIAGNOSIS','DECEASED',      
    df.drop(columns=['HADM_ID', 'ADMITTIME', 'ADMISSION_LOCATION',
                'DISCHARGE_LOCATION', 'LANGUAGE', 'ADMIT_MIN', 'DOD',
                'DEATHTIME'], inplace=True)
    
    prefix_cols = ['ADM', 'INS', 'ETH', 'AGE', 'MAR'] # 'REL',  'RELIGION',
    dummy_cols = ['ADMISSION_TYPE', 'INSURANCE', 
                 'ETHNICITY', 'age', 'MARITAL_STATUS']
    df = pd.get_dummies(df, prefix=prefix_cols, columns=dummy_cols)
    
    if verbose: 
        print('Data Preprocessing complete.')
    
    return df, actual_median_los, actual_mean_los

df_tsdat, actual_median_los, actual_mean_los = mimic_los_cleanup(adm_csv='/home/junde/02_LOS/Testdat/ADMISSIONS.csv', 
                                                                 patients_csv='/home/junde/02_LOS/Testdat/PATIENTS.csv',
                                                                 diagcode_csv='/home/junde/02_LOS/Testdat/DIAGNOSES_ICD.csv', 
                                                                 icu_csv='/home/junde/02_LOS/Testdat/ICUSTAYS.csv')
'''

## Verify
#df_tsdat.to_csv('/home/junde/02_LOS/06_Multi_tasks/df_tsdat.csv', index=False)
df_tsdat = pd.read_csv('/home/junde/02_LOS/06_Multi_tasks/df_tsdat.csv')

df_tsdat.drop_duplicates(subset=['SUBJECT_ID','LOS'],keep='first',inplace=True)
df_tsdat = df_tsdat.drop(columns=['DECEASED'])
df_tsfeat = df_tsdat[0:400000]
df_test = df_tsdat[400000:]
#------------------smote算法，比例小非均衡样本,均衡------------------------------------------------------------------
#-------------------------------------------------------------------------- 

# df_tsfeat['y'] = (df_tsfeat['LOS'] > 4).astype(int)
# # train_lbl = df_clean.drop(columns=['LOS','SUBJECT_ID','Unnamed: 15'])
train_lbl = df_tsfeat.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
sw_lbl=train_lbl[(train_lbl['HOSPITAL_EXPIRE_FLAG']==1)]
sw_lbl_data=sw_lbl.values
nork_lbl=train_lbl[(train_lbl['HOSPITAL_EXPIRE_FLAG']==0)]
mult=int(len(nork_lbl)/len(sw_lbl))      #扩充多少倍                      
ex_sw_col=pd.DataFrame(sw_lbl_data.repeat(mult,axis=0))
ex_sw_col.columns=sw_lbl.columns

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic
        
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

sw_lbl_slt=sw_lbl.iloc[:,3:sw_lbl.columns.size-1]
sw_lbl_slt=sw_lbl_slt.astype('float')
sw_lbl_slt_value=sw_lbl_slt.values


#------------------smote，对比例小非均衡类别,扩充,均衡------------------------------------------------------------------
#--------------------------------------------------------------------------
s_sw_lbl=Smote(sw_lbl_slt_value,N=mult*100) 
ex_sw_dat=pd.DataFrame(s_sw_lbl.over_sampling())
ex_sw_lbl=pd.merge(ex_sw_col.iloc[:,0:3],ex_sw_dat,left_index=True,right_index=True,how='right') 
ex_sw_lbl['y']=1
ex_sw_lbl.columns=sw_lbl.columns
df_clean=train_lbl.append([ex_sw_lbl]).reset_index(drop=True)
#--------------------------------------------------------------------------

y1 = df_clean['LOS']
y2 = df_clean['HOSPITAL_EXPIRE_FLAG']
features = df_clean.drop(columns=['SUBJECT_ID','Unnamed: 15','HOSPITAL_EXPIRE_FLAG', 'LOS'])

y1_tst = df_test['LOS']
y2_tst = df_test['HOSPITAL_EXPIRE_FLAG']
features_tst = df_test.drop(columns=['SUBJECT_ID', 'Unnamed: 15','HOSPITAL_EXPIRE_FLAG', 'LOS'])
# y = (y > 4).astype(int)
# data_full = df_clean.join(y, rsuffix='_long_stay')
# data_full.groupby('LOS_long_stay').size().plot.bar()

# Split into train 80% and test 20%
X_train, X_val, y1_train, y1_val, y2_train, y2_val  = train_test_split(features, y1,  y2, test_size=0.3, random_state=42)


from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, GlobalAveragePooling1D, Convolution1D, UpSampling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.layers.core import Flatten

def pyramid_feature_maps(input_layer):
    # pyramid pooling module
    # red   padding='causal',
    red = AveragePooling1D(pool_size=2,name='red_pool')(input_layer)
    red = Convolution1D(filters=64,kernel_size=1,dilation_rate=2,padding='causal',name='red_1_by_1')(red)
    red = UpSampling1D(size=2,name='red_upsampling')(red)
    # yellow
    yellow = AveragePooling1D(pool_size=2,name='yellow_pool')(input_layer)
    yellow = Convolution1D(filters=64,kernel_size=1,dilation_rate=2,padding='causal',name='yellow_1_by_1')(yellow)
    yellow = UpSampling1D(size=2,name='yellow_upsampling')(yellow)
    # blue
    blue = AveragePooling1D(pool_size=2,name='blue_pool')(input_layer)
    blue = Convolution1D(filters=64,kernel_size=1,dilation_rate=2,padding='causal',name='blue_1_by_1')(blue)
    blue = UpSampling1D(size=2,name='blue_upsampling')(blue)
    # green
    green = AveragePooling1D(pool_size=2,name='green_pool')(input_layer)
    green = Convolution1D(filters=64,kernel_size=1,dilation_rate=2,padding='causal',name='green_1_by_1')(green)
    green = UpSampling1D(size=2,name='green_upsampling')(green)
    # base + red + yellow + blue + green
    return keras.layers.concatenate([red,yellow,blue,green])


# 扩充维度，使数据集满足一维卷积的形式 
train_data = X_train.values
val_data = X_val.values
test_data = features_tst.values

x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)  
x_val =val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
x_test = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)

# ont_hot code
y2_train = np_utils.to_categorical(y2_train)
y2_val= np_utils.to_categorical(y2_val)
y2_test = np_utils.to_categorical(y2_tst)

input_img = Input(shape=(51, 1))  # 一维卷积输入层，神经元个数为400（特征数）


from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):  
    # Bottleneck layers
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(bn_size*nb_filter, 1, padding='same')(x)
    
    # Composite function
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(nb_filter, 3, padding='same')(x)
    
    if drop_rate: x = Dropout(drop_rate)(x)
    
    return x
 
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=2)
        
    return x
    
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    
    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv1D(nb_filter, 1,  padding='same')(x)
    if is_max != 0: x = MaxPooling1D(2, padding='same')(x)
    else: x = AveragePooling1D(2, padding='same')(x)
    
    return x
 
growth_rate = 1

'''
x1 = Conv1D(32, 3, activation='relu', padding='same')(input_img)   #卷积层，32个核，宽度为5，激活函数为relu
x1 = Conv1D(32, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)   #  规范层，加速模型收敛，防止过拟合
x1 = MaxPooling1D(2, padding='same')(x1)   # 池化层
# x1 = Attention()(x1)

x1 = Conv1D(64, 1, activation='relu', padding='same')(x1)
x1 = Conv1D(64, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(2, padding='same')(x1)
# x1 = Attention()(x1)

x1 = Conv1D(128, 3, activation='relu', padding='same')(x1)
x1 = Conv1D(128, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(2, padding='same')(x1)

x1 = Conv1D(64, 1, activation='relu', padding='same')(x1)
x1 = Conv1D(64, 5, activation='relu', padding='same')(x1)
x1 = MaxPooling1D(2, padding='same')(x1)

x1 = DenseBlock(x1, 12, growth_rate, drop_rate=0.1)
# x1 = TransitionLayer(x1)
x1 = BatchNormalization()(x1)
# x1 = Attention()(x1)

# encoded = GlobalAveragePooling1D()(x1)
encoded = MaxPooling1D(2, padding='same')(x1)  # 全连接层
encoded = Flatten()(encoded)
'''


x1 = Conv1D(32, 3, activation='relu', padding='same')(input_img)   #卷积层，32个核，宽度为5，激活函数为relu
x1 = Conv1D(32, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)   #  规范层，加速模型收敛，防止过拟合
x1 = MaxPooling1D(2, padding='same')(x1)   # 池化层
x1 = Conv1D(64, 1, activation='relu', padding='same')(x1)
x1 = Conv1D(64, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(2, padding='same')(x1)
x1 = Conv1D(128, 3, activation='relu', padding='same')(x1)
x1 = Conv1D(128, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling1D(2, padding='same')(x1)
x1 = Conv1D(64, 1, activation='relu', padding='same')(x1)
x1 = Conv1D(64, 5, activation='relu', padding='same')(x1)
x1 = BatchNormalization(axis = -1)(x1)

encoded = MaxPooling1D(2, padding='same')(x1) 
#decoded = Dense(128, activation='linear', kernel_regularizer=regularizers.l2(0.1))(encoded)   # Regression问题  linear激活函数，输出层
encoded = Flatten()(encoded)



out_reg = Dense(1, activation='linear', name='reg')(encoded)
out_class = Dense(2, activation='softmax', name='class')(encoded) # I suppose bivariate classification problem
model = Model(input_img, [out_reg, out_class])

learning_rate = 0.001
decay = 1e-6
momentum = 0.9
nesterov = True
sgd_optimizer = SGD(lr = learning_rate, decay = decay, momentum = momentum, nesterov = nesterov)
# model.compile('adam', loss={'reg':'mse', 'class':'binary_crossentropy'}, 
model.compile(optimizer=optimizers.Adam(lr=0.001), loss={'reg':'mse', 'class':[focal_loss(gamma=2)]}, metrics={'reg':[coeff_determination], 'class':['accuracy']}, 
              loss_weights={'reg':0.5, 'class':0.5})
#'categorical_crossentropy'


model.fit(
    x=x_train,
    y=[y1_train,y2_train],
    validation_data=(x_val, [y1_val,y2_val]),
    batch_size=128,
    epochs=30
)



#----统计指标3，testset---------------------------------------------------------------------
y_test_preds = model.predict(x_test)
pre_test = [np.argmax(ii) for ii in y_test_preds[1]]
act_tstlbl = np.array(y2_tst)
tst_acc=accuracy_score(act_tstlbl, pre_test )
print('tst_class_acc='+str(tst_acc))

act_tst_los=y1_tst.values
reg_metric(act_tst_los,  y_test_preds[0])






