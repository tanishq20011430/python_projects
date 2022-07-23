#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[2]:


df=pd.read_csv("lung_cancer_data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.isna().any()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df['GENDER'].value_counts()


# In[9]:


dummy = pd.get_dummies(df['GENDER'])


# In[10]:


df2=pd.concat((df,dummy),axis=1)


# In[11]:


df2.head(2)


# In[12]:


df=df2.drop(columns='GENDER')


# In[13]:


df.head(2)


# In[14]:


df=df.drop(columns='F')


# In[15]:


df.head(2)


# In[16]:


df=df.rename(columns={'M':'GENDER'})


# In[17]:


df.head(2)


# In[18]:


plt.figure(figsize=(10,5))
df['LUNG_CANCER'].value_counts()
sns.countplot(x='LUNG_CANCER',data=df)


# In[19]:


plt.figure(figsize=(10,5))
sns.countplot('GENDER',data=df,hue='LUNG_CANCER')
plt.show()


# In[20]:


plt.figure(figsize=(10,5))
sns.histplot(df['AGE'])


# In[21]:


plt.figure(figsize=(15,10))
hm = sns.heatmap(df.corr(), annot = True)


# In[22]:


x=df.drop(columns='LUNG_CANCER')
y=df['LUNG_CANCER']


# In[23]:


# balancing imbalanced data
from imblearn.under_sampling import NearMiss


# In[24]:


nm=NearMiss()
x_res,y_res=nm.fit_resample(x,y)


# In[25]:


x_res.shape


# In[26]:


y_res.shape


# In[27]:


x.head(2)


# In[28]:


y.head(2)


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(x_res,y_res,test_size=0.1,random_state=9)


# In[30]:


model=LogisticRegression()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


y_pred=model.predict(x_test)


# In[33]:


compare= pd.DataFrame({"Actual Class" : y_test, "Predicted Class": y_pred})
compare.head()


# In[34]:


print(confusion_matrix(y_test, y_test))


# In[35]:


sns.heatmap(confusion_matrix(y_test, y_test), annot = True);


# In[36]:


accuracy_score(y_test,y_pred)


# In[37]:


print('Error Metrics\n\n', classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




