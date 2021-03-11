# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:58:36 2021

@author: hangc
"""
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from apscheduler.schedulers.background import BackgroundScheduler
import time
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import joblib

pollution = pd.read_csv('D:/work/Neo4j/裕华可训练总表.csv',encoding='utf-8',header=0)##读取原表
pollution = pollution.dropna()#剔除NA值
df1 =pollution.sample(n=50000,random_state=None)####先取50000条
df_sample = df1.drop(['排放源清单类型','排放源名称'],axis=1).reset_index()
df_sample = df_sample.drop(['index'],axis=1)#去除序号列
#把非数值型变量标签化
df_trans = pd.DataFrame()
df_cat = pd.DataFrame()
features = ['输送物料','配套措施是否满足要求','现场治理效果是否满足无可见烟尘外溢要求']
for i in features:
    df_cat['{}_cat'.format(i)] = df_sample[i].astype('category').copy()
    df_cat['{}_cat'.format(i)] = df_cat['{}_cat'.format(i)].cat.codes
df_cat['rate'] = df_sample['rate']#把同步率列加上
df_cat['avg_tdc'] = df_sample['avg_tdc']#加入tsp数值列
    
for feature in features:
    le = LabelEncoder()
    le = le.fit(df_sample[feature])
    df_trans[feature] = le.transform(df_sample[feature])
df_trans['rate'] = df_sample['rate']#把同步率列加上
df_trans['avg_tdc'] = df_sample['avg_tdc']#加入tsp数值列
train_set, test_set = train_test_split(df_cat, test_size=0.2, random_state=42)#分训练测试集
X_train = train_set.drop(['avg_tdc'],axis=1)
# Saving feature names for later use
feature_list = list(X_train.columns)
X_test = test_set.drop(['avg_tdc'],axis=1)
y_train = (train_set.avg_tdc <300 )#小于300为TRUE
y_test = (test_set.avg_tdc > 300 )
#随机森林模型建立
rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
filename = 'rf_model.sav' #设置model名称
joblib.dump(rf, filename) #存储模型
# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# list of x locations for plotting
x_values = list(range(len(importances)))
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


















