#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 05:43:54 2018

@author: arvy
"""

import pandas as pd
import numpy as np

data=pd.read_csv('/home/arvy/Documents/ML/datasets/HR_comma_sep.csv')

from sklearn.preprocessing import LabelEncoder
gle=LabelEncoder()
gnere=gle.fit_transform(data.sales)
f=gle.fit_transform(data.salary)
data.sales=gnere
data.salary=f

from sklearn.cross_validation import train_test_split
y=data.left
X=data.drop(['left'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.33, random_state=42)


from sklearn.svm import SVC
model=SVC(C=10)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


