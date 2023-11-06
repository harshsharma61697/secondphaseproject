#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


data=pd.read_csv("TITANIC_TRAIN.CSV")


# In[3]:


data


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.tail()


# In[16]:


data.isnull().tail()


# In[17]:


data.isnull().sum()


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
import math 


# In[8]:


sns.countplot(x="Survived",data=data)


# In[9]:


sns.countplot(x="Survived",hue='Sex',data=data)


# In[10]:


sns.countplot(x='Survived',hue='Age',data=data)


# In[11]:


sns.countplot(x="Survived",hue='Pclass',data=data)


# In[20]:


data['Age'].plot.hist()


# In[22]:


data['Pclass'].plot.hist()


# In[23]:


data['Survived'].plot.hist()


# In[24]:


data['Fare'].plot.hist()


# In[28]:


sns.countplot(x='SibSp',data=data)


# In[33]:


data['Parch'].plot.hist()


# In[35]:


sns.heatmap(data.isnull())


# In[36]:


sns.barplot(x='Pclass',y='Age',data=data)


# In[37]:


sns.boxplot(x='Pclass',y='Age',data=data)


# In[38]:


data.drop('Cabin',axis=1,inplace=True)


# In[39]:


data.head()


# In[40]:


data.dropna(inplace=True)


# In[41]:


data.isnull()


# In[42]:


data.isnull().sum()


# In[43]:


sns.heatmap(data.isnull())


# In[45]:


data['Sex'].head()


# In[49]:


data['Sex'].tail()


# In[59]:


GENDER=pd.get_dummies(data['Sex'],drop_first=True)


# In[58]:


EMBD=pd.get_dummies(data['Embarked'],drop_first=True)


# In[60]:


ptl=pd.get_dummies(data['Pclass'],drop_first=True)


# In[64]:


data.head(2)


# In[66]:


df=pd.concat([data,GENDER,EMBD,ptl],axis=1)


# In[67]:


df.head(3)


# In[69]:


df.drop(['Name','PassengerId','Ticket','Embarked','Pclass','Sex'],axis=1,inplace=True)


# In[70]:


df.head(3)


# In[71]:


x=df.drop('Survived',axis=1)


# In[72]:


x


# In[73]:


y=df['Survived']


# In[74]:


y


# In[114]:


from sklearn.linear_model import LogisticRegression


# In[115]:


from sklearn.model_selection import train_test_split


# In[116]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=4)


# In[117]:


x_train.shape


# In[118]:


y_train.shape


# In[120]:


lm=LogisticRegression()


# In[121]:


lm.fit(x_train,y_train)


# In[122]:


pred=lm.predict(x_test)


# In[129]:


from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[127]:


accuracy_score(y_test,pred)


# In[126]:


classification_report(y_test,pred)


# In[130]:


confusion_matrix(y_test,pred)


# In[ ]:




