# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:42:36 2021

@author: hangc
"""

import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from apscheduler.schedulers.background import BackgroundScheduler
import time
import numpy as np
from sklearn.model_selection import train_test_split
import os
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import socket

class database():
    def __init__(self,ip,user,passwd,dbname):
        self.conn = pymysql.connect(host=ip,  # 此处必须是是127.0.0.1
                        user=user,  # 数据库用户名
                        passwd=passwd,  # 数据库密码
                        db=dbname # 数据库名称
                        )
        self.cursor = self.conn.cursor()
        
    
    def getData(self,sql):
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        columnDes = self.cursor.description
        colName = [columnDes[i][0] for i in range(len(columnDes))]#获取数据库表中表头名称
        df = pd.DataFrame([list(i) for i in data],columns=colName)#表头写入DF
        return df
    
    
class App:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()
    
    def delete(self):
        with self.driver.session() as session:
            session.run("match (n) detach delete n")

    def create_emission(self):
        with self.driver.session() as session:
            # Write transactions allow the driver to handle retries and transient errors
            session.write_transaction(self._create_emission)
            # for record in result:
            #     print("Created friendship between: {p1}, {p2}".format(
            #         p1=record['p1'], p2=record['p2']))

    @staticmethod
    def _create_emission(tx):
        query1 = (
            "LOAD CSV WITH HEADERS FROM 'file:///emission.csv' AS row "
            "MERGE (:Emission {name: row.排放源名称, id:row.清单ID,category:row.清单属性 })"
        )
        tx.run(query1)
        query2 = (
            '''
            LOAD CSV WITH HEADERS FROM 'file:///relation.csv' AS row 
            MATCH (e1:Emission{id:row.上级排放源编号}),(e2:Emission{id:row.清单ID}) 
            CREATE (e1)-[:forward]->(e2)
            '''
            )
        tx.run(query2)
    
   
    def find_emission(self, emission_id_list):
        with self.driver.session() as session:
            result = list()
            for emission_id in emission_id_list:
                result = result+session.read_transaction(self._find_and_return_emission, emission_id)
            return result

    @staticmethod
    def _find_and_return_emission(tx, emission_id):           
        query = (
        '''
        match (e1:Emission)-[*1..3]->(e2:Emission{id:$emission_id}) 
        return e1,e2
        '''
        )
        result = tx.run(query, emission_id=emission_id)
        emi_list = list()
        for record in result:
            for element in record:
                emi_list.append(element["id"])
        return list(dict.fromkeys(emi_list))

class Model:
    def __init__(self,path):
        self.model = joblib.load(path)
    
    def _labelEncoder(self,df):
        self.boolean_dict = {'是':1,'否':0}
        df = df.drop(['emission_ID'],axis = 1)
        features = list(df.columns)
        df_cat = pd.DataFrame()
        for feature in features:
            le = LabelEncoder()
            le = le.fit(df[feature])
            df_cat[feature] = le.transform(df[feature])
        
        df_dummies = pd.get_dummies(df, columns=['material'], drop_first=True) #转换One-hot
        df_dummies['measures'] = df_dummies['treatment'].map(self.boolean_dict)
        df_dummies['overflow'] = df_dummies['plumeoverflow'].map(self.boolean_dict)
        df_dummies.drop(['treatment','plumeoverflow'],axis=1,inplace=True)
        return df_dummies
    
    def run(self,data):
        test = self._oneHot(data)
        print(list(test))
        result = self.model.predict(test)
        return result
    
     

if __name__ == "__main__":
    ip = '127.0.0.1' #本地ip
    user = 'root' #用户名
    passwd = '19920305' #密码
    dbname = 'product' #数据库名称
    tspDatabase = database(ip,user,passwd,dbname) #初始化数据库
    df_tsp = tspDatabase.getData('select * from tsp_trigger;') #获取报警数据框
    emission_target = df_tsp['emission_id'].values #获取报警关联的排放源点位IDarray
    scheme = "neo4j"  # Connecting to Aura, use the "neo4j+s" URI scheme
    host_name = "localhost" #neo4j本地
    port = 7687 #neo4j端口
    url = "{scheme}://{host_name}:{port}".format(scheme=scheme, 
                                host_name=host_name, port=port) #Neo4J IP
    username = "neo4j" #neo4j用户名
    password = "bme@12345" #neo4j密码
    # os.system(r'D:\neo4j-community-4.1.6\bin\neo4j console') #开启neo4j进程
    app = App(url, username, password) #初始化neo4j
    #app.delete()
    #app.create_emission()
    result = app.find_emission(emission_target) #寻找上级排放源，默认三级
    app.close()
    predDatabase = database(ip,user,passwd,'predict') #初始化预测集数据库
    df_predict = predDatabase.getData('select * from emission_predict WHERE emission_ID in {}' .format(tuple(result) )) #读取图数据库上级排放源点数据
    model_path = 'svm_model.sav' #模型路径
    svm_model = Model(model_path) #初始化训练好的模型
    predict = svm_model.run(df_predict)
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
