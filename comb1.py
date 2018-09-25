# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:45:18 2018

@author: latec
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:40:34 2018

@author: latec
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:04:46 2018

@author: latec
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sklearn
import os
print (sklearn.__version__)
path = "c:/gfk/"
# File Paths


filename_read = os.path.join(path,"training.csv")
df = pd.read_csv(filename_read,na_values=['NA','?'], delimiter=';',decimal=',')

filename_read = os.path.join(path,"validation.csv")
dfv = pd.read_csv(filename_read,na_values=['NA','?'], delimiter=';',decimal=',')

frames = [df, dfv]
result = pd.concat(frames, axis=0)
df=result

df.isnull().sum(axis = 0)
df.isnull().sum(axis = 1)

df1=df.drop(['v17'], axis=1) #too many nans
df1a=df1.dropna() #drop rows with nans
df2=pd.get_dummies(df1a)
print(df2.describe())
#df.describe()

#1463 first rows are from training set
newdf = df2[df2.columns[0:25]]
yall=df2['classLabel_yes.']
y=yall[:1463]
x=newdf[:1463]
x_train,x_test_all,y_train,y_test_all = train_test_split(x,y,test_size = 0.3,random_state=9)

no_trees = 4

estimator = RandomForestClassifier(n_estimators=no_trees)

estimator.fit(x_train,y_train)

train_predicited = estimator.predict(x_train)

train_score = accuracy_score(y_train,train_predicited)

test_predicted = estimator.predict(x_test_all)

test_score = accuracy_score(y_test_all,test_predicted)

print ("Training Accuracy = %0.2f test Accuracy = %0.2f"%(train_score,test_score))

importances= estimator.feature_importances_
validationset=newdf[1463:]
validation_predicted = estimator.predict(validationset)
validation_results=yall[1463:]
test_score = accuracy_score(validation_results,validation_predicted)
print ("Validation Accuracy = %0.2f "%(test_score))

 