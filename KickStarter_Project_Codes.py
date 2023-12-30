#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:21:40 2022

@author: Elaine Yu

Individual Project

"""

import pandas
import numpy

# Import Dataframe
kickstarter_df = pandas.read_excel('/Users/elaineyu/Downloads/Kickstarter.xlsx')

# Pre-Processing
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('canceled', na = False)] #Removing rows containing "canceled", "suspended" and "live"  
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('live', na = False)]
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('suspended', na = False)]

kickstarter_df = kickstarter_df.drop(kickstarter_df.columns[[0,1,3,8,9,10,11,13,15,17,22,23,25,26,27,28,29,30,31,32,33,38,39,40,41,43,44]], axis = 1) # removing variables about after project submission

kickstarter_df = kickstarter_df.dropna() # remove null values

# Dummify categorical variables
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['state','disable_communication', 'country','currency','staff_pick','category',
                                                               'created_at_weekday','created_at_month','created_at_day','created_at_yr','created_at_hr'], drop_first = True)


#%%

# Feature Selection
X = kickstarter_df.drop(columns=['state_successful'])
y = kickstarter_df['state_successful']


# random forest
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 0)
model2 = randomforest.fit(X,y)
model2.feature_importances_

df2 = pandas.DataFrame(list(zip(X.columns,model2.feature_importances_)), columns = ['predictor','feature importance'])
df2.sort_values(['feature importance'], ascending = [False])

# Variables to keep:
    # staff_pick_True, goal, category, name_len, name_len_clean, blurb_len, blurb_len_clean, create_to_launch_days


#%%

# 1. Develop Classification Model

# Setup the variables
X = kickstarter_df[['goal','staff_pick_True','name_len_clean','create_to_launch_days','name_len','blurb_len','blurb_len_clean','category_Apps','category_Blues','category_Comedy',
                    'category_Experimental','category_Festivals','category_Flight','category_Gadgets','category_Hardware','category_Immersive','category_Makerspaces','category_Musical',
                    'category_Places','category_Plays','category_Robots','category_Shorts','category_Software','category_Sound','category_Spaces','category_Thrillers','category_Wearables',
                    'category_Web','category_Webseries']]

y = kickstarter_df["state_successful"]

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# Build the model 
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier()
model3 = randomforest.fit(X_train,y_train)

# Using the model to predict the results based on the test dataset
y_test_pred = model3.predict(X_test)

# Calculate the mean squared error of the prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred)
# 0.7638921453692848

# Confusion Matrix
from sklearn import metrics
metrics.confusion_matrix(y_test,y_test_pred)
print(pandas.DataFrame(metrics.confusion_matrix(y_test,y_test_pred,labels=[0,1]),index=['true:0','true:1'],columns=['pred:0','pred:1']))

# Precision/Recall
metrics.precision_score(y_test,y_test_pred)
metrics.recall_score(y_test,y_test_pred)

# F1 Score
metrics.f1_score(y_test,y_test_pred)

'''
# lr
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model5 = lr.fit(X_train,y_train)
y_test_pred = model5.predict(X_test)
accuracy_score(y_test, y_test_pred)
# 0.6757327080890974
metrics.precision_score(y_test,y_test_pred)
# 0.5444743935309974
metrics.recall_score(y_test,y_test_pred)
# 0.1426553672316384
'''

#%%

# 2. Develop Clustering Model

X = kickstarter_df[['state_successful','goal','name_len_clean','create_to_launch_days','name_len','blurb_len','blurb_len_clean']]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_std = scaler.fit_transform(X)

# Find optimal number of clusters

# Elbow Method
from sklearn.cluster import KMeans
withinss = []
for i in range (1,8):
    kmeans = KMeans(n_clusters = i)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)
    
from matplotlib import pyplot
pyplot.plot([1,2,3,4,5,6,7],withinss)
# n_clusters = 2

# Silhouette Method
kmeans = KMeans(n_clusters = 2)
model4 = kmeans.fit(X_std)
labels = model4.predict(X_std)

from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std,labels)

df = pandas.DataFrame({'label':labels,'silhouette':silhouette})

print('Average Silhouette Score for Cluster 0: ',numpy.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',numpy.average(df[df['label'] == 1].silhouette))


from sklearn.metrics import silhouette_score
silhouette_score(X_std,labels)
# 0.6221653195348583
# provides good evidence of the reality of the clusters in the data

centroids = model.cluster_centers_

X['ClusterMembership'] = labels

cluster_0 = X[X['ClusterMembership'] == 0].mean()
cluster_1 = X[X['ClusterMembership'] == 1].mean()

clusters = pandas.concat([cluster_0,cluster_1], axis = 1)
print(clusters)

#%%

# Grading

# Import Dataframe
kickstarter_df = pandas.read_excel('Kickstarter-Grading.xlsx')

# Pre-Processing
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('canceled', na = False)]  
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('live', na = False)]
kickstarter_df = kickstarter_df[~kickstarter_df['state'].str.contains('suspended', na = False)]

kickstarter_df = kickstarter_df.drop(kickstarter_df.columns[[0,1,3,8,9,10,11,13,15,17,22,23,25,26,27,28,29,30,31,32,33,38,39,40,41,43,44]], axis = 1) 

kickstarter_df = kickstarter_df.dropna() 

# Dummify categorical variables
kickstarter_df = pandas.get_dummies(kickstarter_df, columns = ['state','disable_communication', 'country','currency','staff_pick','category',
                                                               'created_at_weekday','created_at_month','created_at_day','created_at_yr','created_at_hr'], drop_first = True)

# Setup the variables
X_grading = kickstarter_df[['goal','staff_pick_True','name_len_clean','create_to_launch_days','name_len','blurb_len','blurb_len_clean','category_Apps','category_Blues','category_Comedy',
                    'category_Experimental','category_Festivals','category_Flight','category_Gadgets','category_Hardware','category_Immersive','category_Makerspaces','category_Musical',
                    'category_Places','category_Plays','category_Robots','category_Shorts','category_Software','category_Sound','category_Spaces','category_Thrillers','category_Wearables',
                    'category_Web','category_Webseries']]

y_grading = kickstarter_df["state_successful"]

# Apply the model previously trained to the grading data
y_grading_pred = model3.predict(X_grading)

# Calculate the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_grading, y_grading_pred)

