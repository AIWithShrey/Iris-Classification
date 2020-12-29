#!/usr/bin/env python
# coding: utf-8

# In[57]:


from sklearn.datasets import load_iris

iris_dataset = load_iris()


# **The characteristics of the dataset**

# In[58]:


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))


# **A short description of what the iris dataset contains**

# In[59]:


print(iris_dataset['DESCR'][:193] + "\n...")


# **Target names are the labels**

# In[60]:


print("Target names: {}".format(iris_dataset['target_names']))


# **Feature names are basically the features**

# In[61]:


print("Feature names: \n{}".format(iris_dataset['feature_names']))


# **Type of data being stored:**

# In[62]:


print("Type of data: {}".format(type(iris_dataset['data'])))


# **Shape of the array in (rows, columns), 150 samples multiplied with 4 features (the properties of each sample)**

# In[63]:


print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))


# In[64]:


print("Type of target: {}".format(type(iris_dataset['target'])))


# **'target' is also a one-dimensional NumPy array**

# In[65]:


print("Shape of target: {}".format(iris_dataset['target'].shape))


# **'target' is converted from labels (text) to integers between 0 and 2. This is called encoding.**

# In[66]:


print("Target:\n{}".format(iris_dataset['target']))


# **Each number corresponds to a label: 0 represents *Setosa*, 1 represents *versicolor*, and 2 represents *virginica*.**
# 

# # Splitting our dataset into training and testing data with the help of Sci-kit learn.

# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)


# **The 'random_state = 0' command helps us set the sequence of the shuffled data to a constant to make the output more deterministic. hence helping us debug the program.**

# **The 'train_test_split()' splits the train data and test data by 75% and 25% correspondingly.**

# In[69]:


print("X_train shape: {}".format(X_train.shape)) 
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape)) 
print("y_test shape: {}".format(y_test.shape))


# # Visualization

# Using Pandas' DataFrame and scatter_matrix() function, we can visualise our data in a scatter plot graph.

# In[70]:


import pandas as pd
from pandas.plotting import scatter_matrix
import mglearn

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)


# In[71]:


grr = scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)


# # The model: K-nearest neighbours

# In[72]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# In[73]:


knn.fit(X_train, y_train)


# In[74]:


import numpy as np

X_new = np.array([[5, 2.9, 1, 0.2]])

print("X_new.shape: {}".format(X_new.shape))


# **Congrats on building your first model. Let us now make predictions using it.**

# # Make a prediction.

# In[75]:


prediction = knn.predict(X_new) 
print("Prediction: {}".format(prediction)) 
print("Predicted target name: {}".format( iris_dataset['target_names'][prediction]))


# In[76]:


y_pred = knn.predict(X_test) 
print("Test set predictions:\n {}".format(y_pred))


# In[77]:


print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




