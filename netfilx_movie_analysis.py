#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


df=pd.read_csv('mymoviedb.csv',lineterminator='\n')
df.head()


# In[30]:


df.info()


# In[31]:


df['Genre'].head()


# In[32]:


df.duplicated().sum()


# In[33]:


df.describe()


# # We have a dataframe consisting of 9,827 rows and 9 columns.
# 
# 

# # Our dataset looks fairly tidy, with no NaNs and no duplicated values.
# 

# # The Release_Date column needs to be converted into datetime format.
# 
# 

# # The Overview, Original_Language, and Poster_URL columns will not be very useful for our analysis and can be excluded.
# 
# 

# # There are noticeable outliers in the Popularity column that need to be handled.
# 
# 

# # The Vote_Average column should be categorized for more meaningful analysis.
# 
# 

# # The Genre column contains comma-separated values and extra white spaces that need to be cleaned.

# In[34]:


df['Release_Date']=pd.to_datetime(df['Release_Date'])
print(df['Release_Date'].dtypes)


# In[35]:


df['Release_Date']=df['Release_Date'].dt.year
df['Release_Date'].dtypes


# In[36]:


df.head()


# # dropping the columns

# In[38]:


cols=['Overview', 'Original_Language', 'Poster_Url']
df.drop(cols, axis=1, inplace=True)
df.columns


# In[39]:


df.head()


# # categorize vote_average column

# In[46]:


def categorize_col(df,col,labels):
    edges=df[col].quantile([0,0.25,0.5,0.75,1]).tolist()
    df[col]=pd.cut(df[col],edges,labels=labels,duplicates='drop') 
    return df


# In[47]:


#define labels for edges
labels=['not_popular','below_average','average','popular']
#categorize column based on labels and edges
categorize_col(df,'Vote_Average',labels)
#confirming changes
df['Vote_Average'].unique()


# In[48]:


df.head()


# In[50]:


#exploring column
df['Vote_Average'].value_counts()


# In[53]:


#drops nans
df.dropna(inplace=True)
#confirming
df.isna().sum()


# In[54]:


df.head()


# # we had split into list and then explode our dataframe to have only one genre per row for each movie

# In[55]:


#split the string into lists
df['Genre']=df['Genre'].str.split(', ')
#explode the lists
df=df.explode('Genre').reset_index(drop=True)
df.head()


# In[57]:


#casting column into category
df['Genre']=df['Genre'].astype('category')
# confirming changes
df['Genre'].dtypes


# In[58]:


df.info()


# In[59]:


df.nunique()


# # Data Visualization

# In[64]:


#setting up seaborn configurations
sns.set_style('whitegrid')


# # Q1? what is the most ferquent genre in the dataset

# In[65]:


#showing stats,on green column
df['Genre'].describe()


# # visualizing genre column
# sns.catplot(y='Genre',data=df,kind='count',order=df['Genre'].value_counts().index,color='#42875f')
# plt.title('genre column distribution')
# plt.show()

# In[68]:


sns.catplot(y='Genre',data=df,kind='count',order=df['Genre'].value_counts().index,color='#42875f') 
plt.title('genre column distribution') 
plt.show()


# # Q2? what genres has highest votes?

# In[69]:


#visualizing vote_average column
sns.catplot(y='Vote_Average',data=df,kind='count',order=df['Vote_Average'].value_counts().index,color='#42785f')
plt.title('votes Distribution')
plt.show()


# # Q3? what movie the highest popularity? what's its gentre?

# In[ ]:


# checking max popularity in dataset
df[df['Popularity']==df['Popularity'].max()]


# # Q3? what movie the lowest popularity? what's its gentre?
# 

# In[73]:


# checking min popularity in dataset
df[df['Popularity']==df['Popularity'].min()]


# # Q5? which year has the most filmmed movies?

# In[77]:


df['Release_Date'].hist()
plt.title('Release_date column distribution')
plt.show()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'dataset')

Drama genre is the most frequent genre in our dataset and has appeared more than
14% of the times among 19 other genres.

Q2: What genres has highest votes ?

we have 25.5% of our dataset with popular vote (6520 rows). Drama again gets the
highest popularity among fans by being having more than 18.5% of movies popularities.

Q3: What movie got the highest popularity ? what's its genre ?

Spider-Man: No Way Home has the highest popularity rate in our dataset and it has
genres of Action, Adventure and Sience Fiction .

Q3: What movie got the lowest popularity ? what's its genre ?

The united states, thread' has the highest lowest rate in our dataset
and it has genres of music , drama , 'war', 'sci-fi' and history'.

Q4: Which year has the most filmmed movies?

year 2020 has the highest filmming rate in our dataset.


# In[ ]:




