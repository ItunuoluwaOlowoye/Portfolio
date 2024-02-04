#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:36:19 2021

@author: haythamomar
"""

import pandas as pd

import numpy as np

retail= pd.read_csv('retail_clean.csv')
rfm= pd.read_csv('rfm_revised.csv')

### splitting the columns
rfm['recency_groups']= rfm['rec_freq_monet'].astype('string').str.slice(0,1,1)
rfm['frequency_groups']= rfm['rec_freq_monet'].astype('string').str.slice(1,2,1)
rfm['monetary_groups']= rfm['rec_freq_monet'].astype('string').str.slice(2,3,1)

value_map= {'1':'3','3':'1','2':'2'}


rfm['recency_groups']= rfm['recency_groups'].map(value_map)
rfm['frequency_groups']= rfm['frequency_groups'].map(value_map)
rfm['monetary_groups']= rfm['monetary_groups'].map(value_map)


rfm['overall_score']= (rfm['recency_groups'].astype('int64')+rfm['frequency_groups'].astype('int64')+
                       rfm['monetary_groups'].astype('int64'))

### getting life time value
ltv= retail.groupby('Customer ID')['Revenue'].sum().reset_index()
ltv.columns=['Customer ID','ltv']
import matplotlib.pyplot as plt

ltv.ltv.plot(kind='hist')

import seaborn as sns


sns.boxplot(y='ltv',data=ltv)

outliers_removed= ltv[ltv.ltv <= ltv.ltv.quantile(0.99)]
len(ltv)-len(outliers_removed)

sns.boxplot(y='ltv',data=outliers_removed)


from sklearn.cluster import KMeans

km= KMeans(n_clusters=3,n_init=10,max_iter=300)

fitting= km.fit_predict(outliers_removed[['ltv']])

outliers_removed['clusters']=fitting

outliers_removed.groupby('clusters')['ltv'].mean()

outliers_removed['clusters']=outliers_removed['clusters'].astype('string')

ltv_mapping= {'0':'Low_ltv','1': 'Mid_ltv','2':'High_ltv'}


outliers_removed['clusters']=outliers_removed['clusters'].map(ltv_mapping)


rfm.columns

rfm_data= rfm.loc[:,['Customer ID','recency','frequency','monetary','recency_groups',
                     'frequency_groups', 'monetary_groups','overall_score']]


joined_data= pd.merge(outliers_removed[['Customer ID','clusters']],rfm_data,
                      how='left',on= 'Customer ID')


joined_data.drop('Customer ID',axis=1,inplace=True)

X_1= pd.get_dummies(joined_data.drop('clusters',axis=1))
columns=X_1.columns
X=X_1.values

y= joined_data['clusters'].values

print(X_1)
print(y)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

model_tree= DecisionTreeClassifier()


cv= RepeatedStratifiedKFold(n_splits=3,n_repeats=3,random_state=1)

scores= cross_val_score(model_tree,X,y,scoring='accuracy',cv=cv)

print(scores.mean())


param_dist = {"max_depth": [3, None],
          
           "min_samples_leaf": range(1,9),
             "criterion": ["gini", "entropy"]}


tree= DecisionTreeClassifier()
rf= RandomForestClassifier()

tree_cv= RandomizedSearchCV(tree,param_dist,cv=5)
rf_cv= RandomizedSearchCV(rf, param_dist,cv=5)


tree_cv.fit(X,y)

rf_cv.fit(X,y)

tree_cv.best_score_

rf_cv.best_score_


prediction=rf_cv.predict(X)


comparison_data= pd.DataFrame({'Actual': y,'Prediction':prediction})


comparison_data.groupby(['Actual','Prediction'])['Actual','Prediction'].count()






























