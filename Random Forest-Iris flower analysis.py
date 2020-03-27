# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 12:00:37 2020

@author: Ishan
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
np.random.seed(0)

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris)

df["Species"]=pd.Categorical.from_codes(iris.target,iris.target_names)
print(df.head())

df["is_train"]=np.random.uniform(0,1,len(df)) <=0.75
print(df.head())

train,test=df[df["is_train"]==True],df[df["is_train"]==False]
print(len(train))
print(len(test))

features=df.columns[:4]
y=pd.factorize(train["Species"])[0]
print(y)

Randomforest=RandomForestClassifier(n_jobs=2,random_state=0)
Randomforest.fit(train[features],y)
Randomforest.predict(test[features])
Randomforest.predict_proba(test[features])[10:20]

preds=iris.target_names[Randomforest.predict(test[features])]
preds[0:25]

pd.crosstab(test.Species,preds,rownames=["Actual"],colnames=["Predicted"])



















