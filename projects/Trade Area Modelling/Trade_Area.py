#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 15:37:23 2021

@author: haythamomar
"""
import pandas as pd
import numpy as np

xls= pd.ExcelFile('Python_trade.xlsx')

xls.sheet_names

distance= xls.parse(0,index_col=0)
census= xls.parse(1,index_col=0)
census['total']= census['Households']*census['Expenditure grocery']

attractivness= xls.parse(2,index_col=0)

stores= np.unique(attractivness.index)
commu= np.unique(distance.index)

keys= [(x,y) for x in commu for y in stores]

#### we will calucluate the numinator and we scale the attractivness

scaled= (attractivness- attractivness.min())/(attractivness.max()-attractivness.min())

scaled.columns
scaled['attractivness']=(scaled['Size']+ scaled['Parking spaces']+scaled['Highways']+
scaled['Traffic']+scaled['Accessibility']+scaled['Design'])

neum= {}

for key in keys:
    neum[key]= scaled.loc[key[1],'attractivness']/distance.loc[key[0],key[1]]**2

neum

##numerator

Pijs={}

for key in keys:
    Pijs[key]= neum[key]/sum([v for k,v in neum.items() if k[0]== key[0]])


### expected_per_key

exp_key={}
census
for key in keys:
    exp_key[key]= Pijs[key]* census.loc[key[0],'total']
    
exp_store= {}    
    
for store in stores:
    exp_store[store]= sum([v for k,v in exp_key.items() if k[1]==store])
     
exp_store   









