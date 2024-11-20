# -*- coding: utf-8 -*-
"""
@author: S.Sarkar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold
import random

def f(gender):
    if 'female' in gender:
        return 0
    elif 'male' in gender:
        return 1

def g(Overall_Laterality):
      if 'L' in Overall_Laterality:
        return 0
      elif 'R' in Overall_Laterality:
        return 1
      elif 'UL' in Overall_Laterality:
        return 0
      elif 'UR' in Overall_Laterality:
        return 1
    
dataset = pd.read_csv('D:\Machin Learning\Laterality_Final.csv')
dataset['gender_No'] = dataset['gender'].apply(f)
dataset['Overall_Laterality_NO'] = dataset['Overall_Laterality'].apply(g)

X = dataset.iloc[:, [10,12,14,16,18,23]]
y = dataset.iloc[:, 24]

dataset.Hipp_Vol_LI.fillna(dataset.Hipp_Vol_LI.mean())
dataset.Hipp_FLAIR_LI.fillna(dataset.Hipp_FLAIR_LI.mean())
dataset.Cg_LI.fillna(dataset.Cg_LI.mean())
dataset.Fx_LI.fillna(dataset.Fx_LI.mean())
dataset.Hipp_MD_LI.fillna(dataset.Hipp_MD_LI.mean());

scaler = MinMaxScaler(feature_range =(0,1))
X = scaler.fit_transform(X)
# Create Models
model = AdaBoostClassifier(n_estimators = 200, random_state = 0, algorithm = 'SAMME')
model = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state = 0)
model = SVC(kernel = 'poly', C = 0.005)
model = LogisticRegression(C =10000)
model = KNeighborsClassifier(n_neighbors= 4,weights = 'distance')

sum=0
model_sum= 0
counter = 0
rkf =RepeatedKFold(n_splits=2,n_repeats=10, random_state=42)

for train_index, test_index in rkf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    model.fit(X_train,y_train)
    model_score = model.score(X_train, y_train)
    acc = accuracy_score(y_test,y_pred)
    sum = sum + acc
    model_sum = model_sum + model_score
    counter += 1
avg_score = model_sum/counter
avg_acc = sum/counter
print('avg_scor =',avg_score)
print('avg_acc =',avg_acc)

