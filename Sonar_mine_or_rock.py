#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


sonar = pd.read_csv('sonar.csv',header=None)


# In[3]:


sonar.head()


# In[4]:


sonar.tail()


# In[5]:


sonar.info()


# In[6]:


sonar.describe()


# In[7]:


sonar.shape


# In[8]:


sonar.isna().any()


# In[9]:


sonar[60].value_counts() # M -> Mines
                         # R -> Rocks


# In[10]:


sonar.groupby(60).mean()


# In[11]:


x =sonar.drop(columns=60,axis=1)
y =sonar[60]
print(x)
print(y)


# In[12]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)


# In[13]:


model= LogisticRegression()


# In[14]:


model.fit(x_train,y_train)


# In[15]:


# accuracy on train data
x_pred_train=model.predict(x_train)
x_pred_train_acc=accuracy_score(x_pred_train,y_train)
print("Accuracy on traing data : ",x_pred_train_acc)


# In[16]:


# accuracy on test data
x_pred_test=model.predict(x_test)
x_pred_test_acc=accuracy_score(x_pred_test,y_test)
print("Accuracy on test data : ",x_pred_test_acc)


# In[17]:


# making a predictive system
input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062
)

# changing the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape numpy array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshape)
print(prediction)

if(prediction[0]=='R'):
    print("Its a Rock")
else:
    print("Its a Mine")


# In[ ]:




