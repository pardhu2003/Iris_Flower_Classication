#!/usr/bin/env python
# coding: utf-8

# ## Iris_Flower_Classification

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("IRIS.csv")
data


# In[3]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[18]:


# Features (sepal length, sepal width, petal length, petal width)


# Target (species): Setosa, Versicolor, Virginica

X = data.drop(['species'],axis=1)
y = data['species']
# Print the first few samples
print(X[:5])
print(y[:5])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=123)



# In[19]:


scalar = StandardScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)


# In[23]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)


# In[24]:


y_pred = knn.predict(X_test)


# In[25]:


print(confusion_matrix(y_test,y_pred))


# In[26]:


print(classification_report(y_test,y_pred))


# In[ ]:




