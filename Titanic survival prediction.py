#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train=pd.read_csv('train.csv')


# In[4]:


train


# In[5]:


train.isnull()


# In[6]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[7]:


sns.countplot('Survived',data=train,hue='Sex')


# In[8]:


sns.countplot('Pclass',data=train)


# In[9]:


train.head()


# In[10]:


sns.distplot(train['Age'].dropna(),bins=30,kde=False)


# In[11]:


train['Age'].plot.hist(bins=30)


# In[12]:


train['Fare']


# In[13]:


train['Fare'].hist()


# In[ ]:





# In[14]:


# train['Fare'].iplot(kind='hist',bins=50)


# In[15]:


train['Age'].mean()


# In[16]:


train['Pclass'].mean()


# In[17]:


##average age of pclass 1

(train["Age"][train['Pclass']==1].mean())


# In[18]:


#average age of pclass 2
(train['Age'][train['Pclass']==2].mean())


# In[19]:


##average age of pclass 3
(train['Age'][train['Pclass']==3].mean())


# In[20]:


sns.set_style('whitegrid')


# In[21]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train, )


# In[22]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 38
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
            return Age
    
    


# In[23]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:





# In[24]:


# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[ ]:





# In[25]:


train.drop('Cabin',axis=1,inplace=True)


# In[26]:


train.head()


# In[27]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[28]:


train.dropna(inplace=True)#there is no missing value


# In[29]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False)


# In[30]:


pd.get_dummies(train['Sex'])


# In[31]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[32]:


sex


# In[33]:


embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[34]:


embark


# In[35]:


train=pd.concat([train,sex,embark],axis=1)


# In[36]:


train.head()


# In[37]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[38]:


train.head()


# In[39]:


train.drop('PassengerId',axis=1,inplace=True)


# In[40]:


train.head()


# In[41]:


X = train.drop('Survived',axis=1)
y = train['Survived']


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[44]:


from sklearn.linear_model import LogisticRegression


# In[45]:


logmodel=LogisticRegression()


# In[46]:


logmodel.fit(X_train,y_train)


# In[47]:


predict=logmodel.predict(X_test)


# In[48]:


from sklearn.metrics import confusion_matrix


# In[49]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# In[50]:


from sklearn.metrics import classification_report


# In[51]:


print(classification_report(y_test,predict))


# In[52]:


y_test


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




