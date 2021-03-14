#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


insurance = pd.read_csv('G:\Data Science\Dataset\insurance.csv')
insurance.head()


# In[3]:


# check null value
insurance.isnull().sum()


# In[4]:


# check datatype for each column
insurance.info()


# In[5]:


insurance.describe()


# In[6]:


insurance.shape


# ### Create dummy variables for object datatype

# In[7]:


insurance_dum = pd.get_dummies(insurance, columns = ['sex', 'smoker', 'region'], drop_first = True)


# In[8]:


# re-order the columns
columns = ['age', 'bmi', 'children', 'sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest', 'charges']
insurance_dum = insurance_dum.reindex(columns = columns)


# In[9]:


insurance_dum.head()


# ### Exploratory Data Analysis

# In[10]:


sns.pairplot(insurance_dum)


# In[11]:


sns.distplot(insurance_dum['charges'])
print('Mean of charges is', insurance_dum['charges'].mean())


# In[12]:


plt.figure(figsize=(18,7))

sns.heatmap(insurance_dum.corr(), annot = True, cmap='coolwarm')


# ### Linear Regression

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[14]:


insurance_x = insurance_dum.drop(['charges'], 1)
insurance_y = insurance_dum['charges']


# In[15]:


print(insurance_x.shape)
insurance_x.head()


# In[16]:


insurance_y.head()


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(insurance_x, insurance_y, test_size = 0.2, random_state = 123) 
print('X_train shape is', X_train.shape)
print('X_test shape is', X_test.shape)
print('y_train shape is', y_train.shape)
print('y_test shape is', y_test.shape)


# In[18]:


lr = LinearRegression()

lr.fit(X_train, y_train)


# In[19]:


print('The intercept is', lr.intercept_)


# In[20]:


print(lr.coef_)


# In[21]:


linear_coef_df = pd.DataFrame(lr.coef_, index = X_train.columns, columns = ['linear_coef'])
linear_coef_df


# In[22]:


# charges prediction
ypred_linear = pd.DataFrame(lr.predict(X_test), columns = ['y_pred_lin'])


# In[23]:


plt.scatter(y_test, ypred_linear)


# #### Evaluation Metrics

# In[24]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[25]:


print(f'Mean Absolute Error: {mean_absolute_error(ypred_linear, y_test)}')
print(f'Mean Squared Error: {mean_squared_error(ypred_linear, y_test)}')
print(f'Root Mean Absolute Error: {np.sqrt(mean_absolute_error(ypred_linear, y_test))}')
print(f'R_squared: {r2_score(ypred_linear, y_test)}')


# ### Let's compare to other regression models

# In[26]:


from sklearn import linear_model


# In[27]:


# Lasso Regression
lm_lasso = linear_model.Lasso()

# Ridge Regression
lm_ridge = linear_model.Ridge()

# Elastic Net Regression
lm_elastic = linear_model.ElasticNet()


# In[28]:


lm_lasso.fit(X_train, y_train)
lm_ridge.fit(X_train, y_train)
lm_elastic.fit(X_train, y_train)


# In[29]:


# Return the intercept for each model
print(f'Lasso model intercept: {lm_lasso.intercept_}')
print(f'Ridge model intercept: {lm_ridge.intercept_}')
print(f'Elastic Net model intercept: {lm_elastic.intercept_}')


# In[30]:


# Return coefficient for each model
print(f'Lasso model coefficient: {lm_lasso.coef_}')
print(f'Ridge model coefficient: {lm_ridge.coef_}')
print(f'Elastic Net model coefficient: {lm_elastic.coef_}')


# In[31]:


lasso_coef_df = pd.DataFrame(lm_lasso.coef_, index = X_train.columns, columns = ['lasso_coef'])
lasso_coef_df


# In[32]:


ridge_coef_df = pd.DataFrame(lm_ridge.coef_, index = X_train.columns, columns = ['ridge_coef'])
ridge_coef_df


# In[33]:


elastic_coef_df = pd.DataFrame(lm_elastic.coef_, index = X_train.columns, columns = ['elastic_coef'])
elastic_coef_df


# In[34]:


coefmod_df = pd.concat([linear_coef_df, lasso_coef_df, ridge_coef_df, elastic_coef_df], axis=1)
coefmod_df


# In[35]:


# charges prediction
ypred_lasso = pd.DataFrame(lm_lasso.predict(X_test), columns = ['y_lasso_pred'])
ypred_ridge = pd.DataFrame(lm_ridge.predict(X_test), columns = ['y_ridge_pred'])
ypred_elastic = pd.DataFrame(lm_elastic.predict(X_test), columns = ['y_elastic_pred'])

ypred_df = pd.concat([ypred_linear, ypred_lasso, ypred_ridge, ypred_elastic], axis=1)
ypred_df.head()


# In[36]:


print(f'Linear Mean Absolute Error: {mean_absolute_error(ypred_linear, y_test)}')
print(f'Lasso Mean Absolute Error: {mean_absolute_error(ypred_lasso, y_test)}')
print(f'Ridge Mean Absolute Error: {mean_absolute_error(ypred_ridge, y_test)}')
print(f'Elastic Net Absolute Error: {mean_absolute_error(ypred_elastic, y_test)}')


# In[37]:


print(f'Linear Mean Squared Error: {mean_squared_error(ypred_linear, y_test)}')
print(f'Lasso Mean Squared Error: {mean_squared_error(ypred_lasso, y_test)}')
print(f'Ridge Mean Squared Error: {mean_squared_error(ypred_ridge, y_test)}')
print(f'Elastic Net Mean Squared Error: {mean_squared_error(ypred_elastic, y_test)}')


# In[38]:


print(f'Linear Root Mean Absolute Error: {np.sqrt(mean_absolute_error(ypred_linear, y_test))}')
print(f'Lasso Root Mean Absolute Error: {np.sqrt(mean_absolute_error(ypred_lasso, y_test))}')
print(f'Ridge Root Mean Absolute Error: {np.sqrt(mean_absolute_error(ypred_ridge, y_test))}')
print(f'Elastic Net Root Mean Absolute Error: {np.sqrt(mean_absolute_error(ypred_elastic, y_test))}')


# In[39]:


# INSERT COMMENT HERE

