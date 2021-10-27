#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


df=pd.read_csv("http://bit.ly/w-data")
df.head(5)


# In[27]:


df.describe()


# In[28]:


df.columns


# In[29]:


df.plot(x='Hours',y='Scores',style='_')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('scores')
plt.show()


# In[30]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[32]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[34]:


y_pred = regressor.predict(X_test) 
print(y_pred)


# In[35]:


df = pd.DataFrame({'actual':y_test,'predicted':y_pred})
df


# In[41]:


hour =[8.62]
own_pr=regressor.predict([hour])
print("No of Hours ={}".format([hour]))
print("Predicted Score = {}".format(own_pr[0]))


# In[ ]:




