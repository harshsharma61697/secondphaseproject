#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 


# In[10]:


df=pd.read_csv("G_glass.csv")


# In[11]:


df


# In[12]:


df.shape


# In[13]:


df.describe()


# In[14]:


df.info()


# In[15]:


df.isnull()


# In[16]:


df.isnull().sum()


# In[17]:


df.columns


# In[18]:


df.isnull().any()


# In[19]:


import matplotlib.pyplot as plt


# In[22]:


df.hist(figsize=(20,12))


# In[27]:


import seaborn as sns


# In[28]:



sns.heatmap(df.corr(),annot=True)


# In[29]:


df['type'].value_counts()


# In[31]:


sns.countplot(x='type',data=df)


# In[32]:


sns.histplot(x='RI',data=df,)


# In[33]:


sns.histplot(x='Na',data=df)


# In[34]:


df.head(2)


# In[35]:


sns.histplot(x='Mg',data=df)


# In[36]:


sns.histplot(x='Al',data=df)


# In[37]:


sns.histplot(x='SI',data=df)


# In[39]:


sns.histplot(x='SI',data=df)


# In[40]:


sns.boxplot(data=df)


# In[41]:


df.head(1)


# In[43]:


sns.scatterplot(x='RI',y='Na',data=df,hue='type')


# In[45]:


x=df.drop('type',axis=1)
x.head()


# In[46]:


y=df['type']
y.head()


# In[47]:


from sklearn.model_selection import train_test_split


# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[56]:


print('X_train:',x_train.shape)
print('Y_train:',y_train.shape)
print('X_test:',x_test.shape)
print('Y_test:',y_test.shape)


# In[73]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[77]:


Lo=LogisticRegression()
Lo


# In[78]:


Lo.fit(x_train,y_train)


# In[61]:


from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[81]:


pred=Lo.predict(x_test)


# In[82]:


pred


# In[83]:


accuracy_score(y_test,pred)


# In[84]:


print(confusion_matrix(y_test,pred))


# In[86]:


print(classification_report(y_test,pred))


# In[88]:


li=LinearRegression()
li


# In[89]:


li.fit(x_train,y_train)


# In[92]:


li.coef_


# In[93]:


li.score(x_train,y_train)


# In[94]:


pred=li.predict(x_test)

print('predicted :',pred)
print('accuracy',y_test)


# In[96]:


from sklearn.metrics import r2_score
print(r2_score(y_test,pred))


# In[ ]:




