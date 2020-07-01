# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 18:05:37 2020

@author: Karan
"""

import numpy as np
import matplotlib.pyplot as pyplot   
import pandas as pd

dataset=pd.read_csv('train.csv')

#Filling null values
null=dataset.isnull().sum(axis=0)
null2={}
for i in range(len(list(null.index))):
    if list(null)[i]>0:
        null2[str(list(null.index)[i])]=list(null)[i]
    
keys=list(null2.keys())
dataset2=dataset[keys]

dataset2.dtypes

mode_fill = ['MasVnrArea','LotFrontage','GarageYrBlt','Electrical']

None_fill= ['MasVnrType','Alley','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtQual',                   
      'FireplaceQu','Fence','GarageCond','GarageFinish','GarageQual','GarageType','PoolQC','MiscFeature']

for i in mode_fill:
    mode=dataset[i].mode()[0]
    dataset[i]=dataset[i].fillna(mode)

for i in None_fill:
    dataset[i]=dataset[i].fillna('None')

print(dataset.isnull().sum().sum())


X=dataset.iloc[:,:-1]
X=X.iloc[:,1:]
y=dataset.iloc[:,-1].values
y=y.reshape(-1,1)

#Creating Dummy variables for categorical variables
obj=[]
for i in X.columns:
    if str(X[i].dtype)=='object':
        obj.append(i)

X=pd.get_dummies(X,columns=obj,drop_first=True)
X=X.values

#Scaling values
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
y=sc_y.fit_transform(y)


#Train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


#Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)

n = len(y_pred)
rmse_lr = np.linalg.norm(y_pred - y_test) / np.sqrt(n)


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor2=RandomForestRegressor(n_estimators=500)
regressor2.fit(X_train,y_train)
y_pred2=regressor2.predict(X_test)


n = len(y_pred)
rmse_rf = np.linalg.norm(y_pred2 - y_test) / np.sqrt(n)


#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor3=DecisionTreeRegressor()
regressor3.fit(X_train,y_train)
y_pred3=regressor3.predict(X_test)

n = len(y_pred)
rmse_dt = np.linalg.norm(y_pred3 - y_test) / np.sqrt(n)


#Ensemble Methods
#1. AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
regressor4=AdaBoostRegressor(regressor3)
regressor4.fit(X_train,y_train)
y_pred4=regressor4.predict(X_test)

n = len(y_pred)
rmse_abr = np.linalg.norm(y_pred4 - y_test) / np.sqrt(n)

#2. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor5=GradientBoostingRegressor(learning_rate=0.001,n_estimators=400)
regressor5.fit(X_train,y_train)
y_pred5=regressor5.predict(X_test)


n = len(y_pred5)
rmse_gb = np.linalg.norm(y_pred5 - y_test) / np.sqrt(n)

#Pricipal Component Analysis to reduce dimensionality in order to efficiently apply SVM
from sklearn.decomposition import PCA

#Calculating ratio of variance covered for 2 to 150 number of features
explained_variance=dict()
for i in range(2,150):
    
    pca=PCA(n_components=i)
    X_train_=pca.fit_transform(X_train)
    X_test_=pca.transform(X_test)
    explained_variance[i]=pca.explained_variance_ratio_.sum()


pca=PCA(n_components=100)
X_train_=pca.fit_transform(X_train)
X_test_=pca.transform(X_test)


#Support Vector Regressor
from sklearn.svm import SVR
regressor=SVR(kernel='rbf',C=0.5)
regressor.fit(X_train_,y_train)
y_pred6=regressor.predict(X_test_)


n = len(y_pred5)
rmse_svr = np.linalg.norm(y_pred6 - y_test) / np.sqrt(n)


#Unscale the predictions to acquire actual predicted house prices 
y_actual=sc_y.inverse_transform(y_pred5)
