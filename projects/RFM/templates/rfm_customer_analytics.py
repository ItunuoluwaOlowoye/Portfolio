# DATACAMP
data = pd.read_csv('some_data.csv')
# Create a spend quartile with 4 groups - a range between 1 and 5
spend_quartile = pd.qcut(data['Spend'], q=4, labels=range(1,5))

# Assign the quartile values to the Spend_Quartile column in data
data['Spend_Quartile'] = spend_quartile

# Print data with sorted Spend values
print(data.sort_values('Spend'))

# Store labels from 4 to 1 in a decreasing order
r_labels = list(range(4, 0, -1))

# Create a spend quartile with 4 groups and pass the previously created labels 
recency_quartiles = pd.qcut(data['Recency_Days'], q=4, labels=r_labels)

# Assign the quartile values to the Recency_Quartile column in `data`
data['Recency_Quartile'] = recency_quartiles 

# Print `data` with sorted Recency_Days values
print(data.sort_values('Recency_Days'))

# Calculate Recency, Frequency and Monetary value for each customer 
datamart = online.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename the columns 
datamart.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

# Print top 5 rows
print(datamart.head())

# Create labels for Recency and Frequency
r_labels = range(3, 0, -1); f_labels = range(1, 4)

# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(datamart['Recency'], q=3, labels=r_labels)

# Assign these labels to three equal percentile groups 
f_groups = pd.qcut(datamart['Frequency'], q=3, labels=f_labels)

# Create new columns R and F 
datamart = datamart.assign(R=r_groups.values, F=f_groups.values)

# Create labels for MonetaryValue
m_labels = range(1, 4)

# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(datamart['MonetaryValue'], q=3, labels=m_labels)

# Create new column M
datamart = datamart.assign(M=m_groups)

# Calculate RFM_Score
datamart['RFM_Score'] = datamart[['R','F','M']].sum(axis=1)
print(datamart['RFM_Score'].head())

# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 10:
        return 'Top'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 10)):
        return 'Middle'
    else:
        return 'Low'

# Create a new variable RFM_Level
datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)

# Print the header with top 5 rows to the console
print(datamart.head())

# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = datamart.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
  
  	# Return the size of each segment
    'MonetaryValue': ['mean', 'count']
}).round(1)

# Print the aggregated dataset
print(rfm_level_agg)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 10:57:57 2020

@author: haythamomar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


retail= pd.read_csv('retail_clean.csv')

retail['InvoiceDate']= pd.to_datetime(retail['InvoiceDate'])
retail['date']= retail['InvoiceDate'].dt.strftime('%Y-%m-%d')
retail['date']= pd.to_datetime(retail['date'])

max_date= retail['date'].max()

retail.columns
customers_recency= retail.groupby('Customer ID').agg(last_date= ('date','max')).reset_index()
customers_recency['recency']= max_date- customers_recency['last_date']
customers_recency['recency']=customers_recency['recency'].astype('string').str.replace('days','').astype(int)

#### frequency

freq2= retail.groupby('Customer ID').date.count().reset_index()
freq2.columns= ['Customer ID','frequency']

freq2






### Monetary value
retail.columns

monet1= retail.groupby(['Customer ID','Invoice']).agg(revenue= ('Revenue','sum')).reset_index()

monet2= monet1.groupby('Customer ID').agg(monetary= ('revenue','mean')).reset_index()


customers_recency['rank_recency']= customers_recency['recency'].rank(pct=True)
freq2['freq_ranking']=freq2['frequency'].rank(ascending=False,pct=True)


monet2['rank_monet']= monet2['monetary'].rank(ascending=False,pct=True)

all_data= pd.merge(customers_recency, freq2,how='left',on='Customer ID')
all_data= pd.merge(all_data,monet2,how='left',on='Customer ID')

bins=[0,0.5,1]
names= ['1','2']

final= pd.DataFrame(customers_recency['Customer ID'])

final['frequency']= pd.cut(freq2['freq_ranking'],bins,labels=names).astype('string')
final['recency']= pd.cut(customers_recency['rank_recency'],bins,labels=names).astype('string')
final['monetary']=pd.cut(monet2['rank_monet'],bins,labels=names).astype('string')

final['rec_freq_mone']=final['recency']+final['frequency']+final['monetary']
print(final['rec_freq_mone'].head())

all_data['rec_freq_monet']= final['rec_freq_mone']

all_data.to_csv('rfm.csv')
import seaborn as sns

fig= sns.countplot(x='rec_freq_monet',data= all_data)

fig.set_xticklabels(fig.get_xticklabels(),
                   rotation=45 )






from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


rfm= pd.read_csv('rfm_revised.csv')
rfm.columns

X= rfm[['frequency','monetary','recency']]
print(X.dtypes)
print(X.head())
km= KMeans(n_clusters=3,n_init= 10,max_iter=300,tol=0.0001)

fitting= km.fit_predict(X)

X['centroids']=fitting

sns.pairplot(data=X,hue='centroids')


sse= []

for k in range(1,11):
    kmeans= KMeans(n_clusters=k,n_init= 10,max_iter=300,tol=0.0001)
    a= kmeans.fit(X)
    sse.append(a.inertia_)
    
    

plt.plot(range(1,11),sse)


























