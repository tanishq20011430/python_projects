#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


churn = pd.read_csv("customer_churn.csv")


# In[3]:


churn.head()


# In[4]:


churn.tail()


# In[5]:


churn.describe()


# In[6]:


churn.shape


# In[7]:


churn.isnull().sum()


# In[8]:


###A) Data Manipulation:
#a. Extract the 5th column & store it in ‘customer_5’
#b. Extract the 15th column & store it in ‘customer_15’
#c. Extract all the male senior citizens whose Payment Method is Electronic check & store the
#result in ‘senior_male_electronic’
#d. Extract all those customers whose tenure is greater than 70 months or their Monthly
#charges is more than 100$ & store the result in ‘customer_total_tenure’
#e. Extract all the customers whose Contract is of two years, payment method is Mailed check
#& the value of Churn is ‘Yes’ & store the result in ‘two_mail_yes’
#f. Extract 333 random records from the customer_churndataframe& store the result in
#‘customer_333’
#g. Get the count of different levels from the ‘Churn’ column


# In[9]:


customer_5=churn.iloc[:4]
print(customer_5)


# In[10]:


customer_15=churn.iloc[13:14]
customer_15


# In[11]:


Senior=churn[(churn['gender']=="Male") & (churn['SeniorCitizen']==1) & (churn['PaymentMethod']=="Electronic check")]


# In[12]:


Senior.head()


# In[13]:


cust_total_tenure=churn[(churn['tenure']>70) | (churn['MonthlyCharges']>100)]


# In[14]:


cust_total_tenure.count()


# In[15]:


two_mail_yes=churn[(churn['Contract']=='Two year')&(churn['PaymentMethod']=="Mailed check")& (churn["Churn"]=="Yes")]


# In[16]:


two_mail_yes.head()


# In[17]:


for col_name in churn:
    print(col_name)


# In[18]:


rand_333=churn.sample(n=333)


# In[19]:


rand_333.head()


# In[20]:


churn["Churn"].value_counts()


# In[21]:


# B) Data Visualization:
#a. Build a bar-plot for the ’InternetService’ column:
#i. Set x-axis label to ‘Categories of Internet Service’
#ii. Set y-axis label to ‘Count of Categories’
#iii. Set the title of plot to be ‘Distribution of Internet Service’
#iv. Set the color of the bars to be ‘orange’
#b. Build a histogram for the ‘tenure’ column:
#i. Set the number of bins to be 30
#ii. Set the color of the bins to be ‘green’
#iii. Assign the title ‘Distribution of tenure’
#c. Build a scatter-plot between ‘MonthlyCharges’ & ‘tenure’. Map ‘MonthlyCharges’ to the yaxis
#& ‘tenure’ to the ‘x-axis’:
#i. Assign the points a color of ‘brown’
#ii. Set the x-axis label to ‘Tenure of customer’
#iii. Set the y-axis label to ‘Monthly Charges of customer’
#iv. Set the title to ‘Tenure vs Monthly Charges’
#d. Build a box-plot between ‘tenure’ & ‘Contract’. Map ‘tenure’ on the y-axis & ‘Contract’ on
#the x-axis.


# In[22]:


plt.figure(figsize=(10,7))
plt.bar(churn['InternetService'].value_counts().keys().tolist(),churn['InternetService'].value_counts().tolist(),color="orange")
plt.xlabel("Categories of Internet Service")
plt.ylabel("Count of Categories")
plt.title("Distribution of Internet Service")
plt.show()


# In[23]:


plt.figure(figsize=(10,7))
plt.hist(churn['tenure'],bins=30,color="green")
plt.title("Distribution of tenure")
plt.show()


# In[24]:


plt.figure(figsize=(15,10))
plt.scatter(x=churn['tenure'],y=churn['MonthlyCharges'],color="brown",marker="o")
plt.xlabel("Tenure of customer")
plt.ylabel("Monthly Charges of customer")
plt.title("Tenure vs Monthly Charges")
plt.show()


# In[25]:


churn.boxplot(column=['tenure'],by=['Contract']);


# In[26]:


#C) Linear Regression:
#a. Build a simple linear model where dependent variable is ‘MonthlyCharges’ and independent
#variable is ‘tenure’
#i. Divide the dataset into train and test sets in 70:30 ratio.
#ii. Build the model on train set and predict the values on test set
#iii. After predicting the values, find the root mean square error
#iv. Find out the error in prediction & store the result in ‘error’
#v. Find the root mean square error


# In[27]:


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[28]:


y=churn[['MonthlyCharges']]
x=churn[['tenure']]


# In[29]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=1)


# In[30]:


x_train.shape,y_train.shape,x_test.shape,y_test.shape


# In[31]:


model = LinearRegression()


# In[32]:


model.fit(x_train,y_train)


# In[33]:


y_pred = model.predict(x_test)


# In[34]:


from sklearn.metrics import mean_squared_error


# In[35]:


np.sqrt(mean_squared_error(y_test,y_pred))


# In[36]:


#D) Logistic Regression:
#a. Build a simple logistic regression modelwhere dependent variable is ‘Churn’ & independent
#variable is ‘MonthlyCharges’
#i. Divide the dataset in 65:35 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and get the accuracy score


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


x=churn[['MonthlyCharges']]
y=churn['Churn']


# In[39]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.35,random_state=1)


# In[40]:


model2 = LogisticRegression()


# In[41]:


model2.fit(x_train,y_train)


# In[42]:


y_pred = model2.predict(x_test)


# In[43]:


from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[44]:


print(metrics.confusion_matrix(y_test, y_pred))


# In[45]:


(1840+0)/(1840+0+626+0)


# In[46]:


accuracy_score(y_test,y_pred)


# In[47]:


#b. Build a multiple logistic regression model where dependent variable is ‘Churn’ &
#independent variables are ‘tenure’ & ‘MonthlyCharges’
#i. Divide the dataset in 80:20 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and get the accuracy score


# In[48]:


x=churn[['MonthlyCharges','tenure']]
y=churn['Churn']


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=1)


# In[50]:


m_model2 = LogisticRegression()


# In[51]:


m_model2.fit(x_train,y_train)


# In[52]:


y_pred=m_model2.predict(x_test)


# In[53]:


y_pred


# In[54]:


accuracy_score(y_test,y_pred)


# In[55]:


print(metrics.confusion_matrix(y_test, y_pred))


# In[56]:


#E) Decision Tree:
#a. Build a decision tree model where dependent variable is ‘Churn’ & independent variable is
#‘tenure’
#i. Divide the dataset in 80:20 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy


# In[57]:


x=churn[['tenure']]
y=churn['Churn']


# In[58]:


from sklearn.tree import DecisionTreeClassifier


# In[59]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=1)


# In[60]:


tree = DecisionTreeClassifier()


# In[61]:


tree.fit(x_train,y_train)


# In[62]:


y_pred = tree.predict(x_test)
y_pred


# In[63]:


print(accuracy_score(y_test,y_pred))


# In[64]:


print(metrics.confusion_matrix(y_test,y_pred))


# In[65]:


#F) Random Forest:
#a. Build a Random Forest model where dependent variable is ‘Churn’ & independent variables
#are ‘tenure’ and ‘MonthlyCharges’
#i. Divide the dataset in 70:30 ratio
#ii. Build the model on train set and predict the values on test set
#iii. Build the confusion matrix and calculate the accuracy


# In[66]:


from sklearn.ensemble import RandomForestClassifier


# In[67]:


x=churn[['tenure']]
y=churn[['Churn']]


# In[68]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.30,random_state=1)


# In[69]:


forest = RandomForestClassifier()


# In[70]:


forest.fit(x_train,y_train)


# In[71]:


y_pred = forest.predict(x_test)


# In[72]:


print(metrics.confusion_matrix(y_test,y_pred))


# In[73]:


accuracy_score(y_test,y_pred)


# In[ ]:




