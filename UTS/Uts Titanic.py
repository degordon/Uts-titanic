#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\dergd\Desktop\UTS\titanic\titanic.csv')
df.head()


# In[2]:


df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()


# In[3]:


target=df.Survived
inputs=df.drop('Survived',axis='columns')


# In[4]:


dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)


# In[5]:


inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)


# In[6]:


inputs.drop('Sex',axis='columns',inplace=True)
inputs.head(3)


# In[7]:


inputs.columns[inputs.isna().any()]


# In[8]:


inputs.Age[:10]


# In[9]:


inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[11]:


len(x_train)


# In[12]:


len(x_test)


# In[13]:


len(inputs)


# In[14]:


from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


# In[15]:


model.fit(x_train, y_train)


# In[16]:


model.score(x_test, y_test)


# In[17]:


x_test[:10]


# In[18]:


y_test[:10]


# In[19]:


model.predict(x_test[:10])


# In[20]:


model.predict_proba(x_test[:10])


# In[ ]:




