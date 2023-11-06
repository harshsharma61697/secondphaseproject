#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd 


# In[10]:


df=pd.read_csv("world.csv")


# In[11]:


df


# In[12]:


df.columns


# In[13]:


df.shape


# In[14]:


df.info()


# In[15]:


df.isnull().sum()


# In[16]:


df.head()


# In[17]:


df.tail()


# In[18]:


df.columns.tolist()


# In[19]:


df.dtypes


# In[20]:


df.isnull().sum().sum()


# In[21]:


import sklearn
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np


# In[22]:


sns.heatmap(df.isnull())


# In[23]:


df.columns.size


# In[24]:


df.index


# In[25]:


df2=pd.read_csv("world.csv",index_col='Country')


# In[26]:


df2


# In[38]:


df3=df2.rename({'Country':"country",'Region':'REGION','Happiness Rank':'Happiness_Rank'})


# In[39]:


df3


# In[27]:


ax=sns.countplot(x='Country',data=df) 
print(df['Country'].value_counts())


# In[ ]:


ax=sns.countplot(x='Happiness Rank',data=df2)
print(df2['Happiness Rank'].value_counts())


# In[ ]:


ax=sns.countplot(x='Dystopia Residual',data=df2)
print(df2['Dystopia Residual'].value_counts())


# In[ ]:


sns.countplot(x='Generosity',data=df2)
print(df2['Generosity'].value_counts())


# In[ ]:


sns.countplot(x='Trust (Government Corruption)',data=df2)
print(df2['Trust (Government Corruption)'].value_counts())


# In[ ]:


sns.barplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df2)


# In[ ]:


sns.scatterplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df2)


# In[ ]:


sns.barplot(x='Region',y='Happiness Rank',data=df2)


# In[ ]:


sns.barplot(x='Trust (Government Corruption)',y='Happiness Rank',data=df2)


# In[ ]:


plt.title('comparison between Trust (Government Corruption) and Happiness Rank')
sns.stripplot(x='Trust (Government Corruption)',y='Happiness Rank',data=df2)
plt.show()


# In[ ]:


df2.hist(bins=25,figsize=(10,10))
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))
sns.pairplot(df2)


# In[ ]:


c=df2['Country'].count()


# In[ ]:




