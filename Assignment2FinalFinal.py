#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Main Libararies
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


#Reading Data
Data=pd.read_csv('assignment2_dataset_cars.csv')
print(Data.describe())


# In[3]:


#Detecting Null Values
Data.isnull().sum()
Data=Data.drop_duplicates(subset=None,keep='first',inplace=False)


# In[4]:


X=Data.iloc[:,0:-1]
Y=Data['price']


# In[9]:


#Extracting Categorical Features
Cat_Data=Data['car_maker']


# In[10]:


#Checking Unique Values
Cat_Data.unique()


# In[11]:


#By Applying Target Encoding 
Categorical_Data=Data.groupby(['car_maker'])['price'].mean().sort_values().index


# In[12]:


# By Identifying Unique Values of car_maker
Categorical_Data


# In[13]:


#By Mapping The Unique Values to a Dictionary
dict1={key:index for index,key in enumerate(Categorical_Data,0)}


# In[14]:


dict1


# In[15]:


# By Applying The New car_maker
Data['car_maker']=Data['car_maker'].map(dict1)
Data['car_maker']


# In[16]:


X['car_maker']=Data['car_maker']
X


# In[17]:


#By Applying Feature Selection by using Correlation Coefficient Method
#Checking Correlation 
Corr=Data.corr()


# In[18]:


#By Checking Top 50% of The Training Features
Top_Feature = Corr.index[abs(Corr['price'])>0.5]
Top_Feature


# In[19]:


#Correlation plot
plt.subplots(figsize=(12, 8))
Top_Corr = Data[Top_Feature].corr()
sns.heatmap(Top_Corr, annot=True)
plt.show()
Top_Feature = Top_Feature.delete(-1)
X = X[Top_Feature]


# In[20]:


#By Splitting The Data and Performing Polynomial Regression
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size=0.8 ,test_size = 0.2,random_state=7,shuffle=True)


# In[21]:


Polynomial_Features = PolynomialFeatures(degree=2)

# transforms the existing features to higher degree features.
X_train_features = Polynomial_Features.fit_transform(X_train)

# fit the transformed features to Linear Regression
Polynomial_Model = linear_model.LinearRegression()
Polynomial_Model.fit(X_train_features, Y_train)

# predicting on training data-set
Y_train_prediction = Polynomial_Model.predict(X_train_features)
Y_Prediction=Polynomial_Model.predict(Polynomial_Features.transform(X_test))

# predicting on test data-set
Prediction = Polynomial_Model.predict(Polynomial_Features.fit_transform(X_test))


# In[22]:


print('Mean Square Error', metrics.mean_squared_error(Y_test, Prediction))


# In[ ]:




