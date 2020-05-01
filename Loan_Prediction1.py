#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_data = pd.read_csv(r'C:\ML\Loan\train_data.csv')


# In[3]:


train_data.head()


# In[4]:


train_data.info()


# In[5]:


train_data.describe()


# In[6]:


sns.countplot(x='Gender',data=train_data)


# In[7]:


train_data['Gender'][train_data['Gender'].isnull()]='Male'


# In[8]:


sns.countplot(x='Married',data=train_data)


# In[9]:


train_data['Married'][train_data['Married'].isnull()]='Yes'


# In[10]:


train_data['LoanAmount'][train_data['LoanAmount'].isnull()]= train_data['LoanAmount'].mean()


# In[11]:


sns.countplot(x='Loan_Amount_Term',data=train_data)


# In[12]:


train_data['Loan_Amount_Term'][train_data['Loan_Amount_Term'].isnull()]='360'


# In[13]:


sns.countplot(x='Self_Employed',data=train_data)


# In[14]:


train_data['Self_Employed'][train_data['Self_Employed'].isnull()]='No'


# In[15]:


sns.countplot(x='Credit_History',data=train_data)


# In[16]:


train_data['Credit_History'][train_data['Credit_History'].isnull()]=1.0


# In[17]:


train_data.info()


# In[18]:


sns.countplot(x='Dependents',data=train_data)


# In[19]:


train_data['Dependents'][train_data['Dependents'].isnull()]='0'


# In[20]:


train_data.loc[train_data.Dependents=='3+','Dependents']= 4


# In[21]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Credit_History');


# In[22]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Gender');


# In[23]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Married');


# In[24]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Dependents');


# In[25]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Education');


# In[26]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Self_Employed');


# In[27]:


grid = sns.FacetGrid(train_data,col='Loan_Status', size=3.2, aspect=1.6)
grid.map(sns.countplot, 'Property_Area');


# In[28]:


train_data.tail()


# In[29]:


train_data.loc[train_data.Loan_Status=='N','Loan_Status']= 0
train_data.loc[train_data.Loan_Status=='Y','Loan_Status']=1


# In[30]:


train_data.loc[train_data.Gender=='Male','Gender']= 0
train_data.loc[train_data.Gender=='Female','Gender']=1


# In[31]:


train_data.loc[train_data.Married=='No','Married']= 0
train_data.loc[train_data.Married=='Yes','Married']=1


# In[32]:


train_data.loc[train_data.Education=='Graduate','Education']= 0
train_data.loc[train_data.Education=='Not Graduate','Education']=1


# In[33]:


train_data.loc[train_data.Self_Employed=='No','Self_Employed']= 0
train_data.loc[train_data.Self_Employed=='Yes','Self_Employed']=1


# In[34]:


property_area= pd.get_dummies(train_data['Property_Area'],drop_first=True)


# In[35]:


train_data= pd.concat([train_data,property_area],axis=1)


# In[36]:


train_data.head()


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X= train_data.drop(['Loan_ID','Property_Area','Loan_Status'],axis=1)
y = train_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[39]:


X['Loan_Amount_Term'] = X.Loan_Amount_Term.astype(int)


# In[40]:


X['new_col'] = X['CoapplicantIncome'] / X['ApplicantIncome']  
X['new_col_2'] = X['LoanAmount'] * X['Loan_Amount_Term']


# In[41]:


X.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount', 'Self_Employed'], axis=1, inplace=True)


# In[42]:


X_train.head()


# In[43]:


X_train['Loan_Amount_Term'] = X_train.Loan_Amount_Term.astype(int)


# In[44]:


X_train['new_col'] = X_train['CoapplicantIncome'] / X_train['ApplicantIncome']  
X_train['new_col_2'] = X_train['LoanAmount'] * X_train['Loan_Amount_Term']


# In[45]:


X_train.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount', 'Self_Employed'], axis=1, inplace=True)


# In[46]:


X_train.head()


# In[47]:


data_corr = pd.concat([X_train, y_train], axis=1)
corr = data_corr.corr()


# In[48]:


plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True);


# In[49]:


X_test['Loan_Amount_Term'] = X_test.Loan_Amount_Term.astype(int)


# In[50]:


X_test['new_col'] = X_test['CoapplicantIncome'] / X_test['ApplicantIncome']  
X_test['new_col_2'] = X_test['LoanAmount'] * X_test['Loan_Amount_Term']


# In[51]:


X_test.drop(['CoapplicantIncome', 'ApplicantIncome', 'Loan_Amount_Term', 'LoanAmount', 'Self_Employed'], axis=1, inplace=True)


# In[52]:


X_test.head()


# In[53]:


X.head()


# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


logmodel = LogisticRegression()


# In[56]:


logmodel.fit(X_train,y_train)


# In[57]:


prediction= logmodel.predict(X_test)


# In[58]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(logmodel,X,y)
print(scores)


# In[59]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)


# In[60]:


from sklearn.metrics import classification_report


# In[61]:


print(classification_report(y_test,prediction))


# In[62]:


X_train.head()


# In[63]:


X_test.head()


# In[64]:


X_test.info()


# In[65]:


X_test_new = X_test.copy()


# In[66]:


X_test_new.info()


# In[67]:


X_test_new.head()


# In[68]:


X_test_new.head()


# In[69]:


accuracy_score(y_test, logmodel.predict(X_test_new))


# In[70]:


print(classification_report(y_test, logmodel.predict(X_test_new)))


# In[ ]:




