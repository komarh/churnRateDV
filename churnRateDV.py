#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("/content/Churn_Modelling.csv")
df.head()


# #### Catploting the dataframe

# In[ ]:


sns.catplot(x="Gender", y="Age", data=df, hue="Exited", height=8, aspect=1.2)


# #### Scatter plots will show the relationship between two numerical variables (estimated salary and balance of customers)

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Estimated Salary vs Balance", fontsize=16)
sns.scatterplot(x="Balance", y="EstimatedSalary", data=df)


# #### Boxplots will show the distribution of a variable in terms of median and quartiles

# In[ ]:


plt.figure(figsize=(12,8))
ax = sns.boxplot(x="Geography", y="Age", data=df)
ax.set_xlabel("Country", fontsize=16)
ax.set_ylabel("Age", fontsize=16)


# #### Distplots will allow us to observe skewness in univariate distribution

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Distribution of Age", fontsize=16)
sns.distplot(df["Age"], hist=False)


# #### Pair plots will give an overview of pairwise relationships among variables

# In[ ]:


subset = df[["CreditScore", "Age", "Balance", "EstimatedSalary"]].sample(n=100)
g = sns.pairplot(subset, height=2.5)


# #### Heatmaps are mostly used to check correlations between features and the target variable

# In[ ]:


corr_matrix = df[['CreditScore','Age','Tenure','Balance',
'EstimatedSalary','Exited']].corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, cmap='Blues_r', annot=True)

