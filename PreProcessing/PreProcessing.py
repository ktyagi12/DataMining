#!/usr/bin/env python
# coding: utf-8


# ###                                           Data Preprocessing
# ###  Question 1: Data Transformation

# In[43]:


import pandas as pd
df = pd.read_csv(r".\specs\SensorData_question1.csv")
print(df)


# ###  Q1 a. Generate a new attribute called Original Input3 which is a copy of the attribute Input3. Do the same with the attribute Input12 and copy it into Original Input12.

# In[44]:


df['Original Input3'] = df['Input3']
df['Original Input12'] = df['Input12']
print(df)


# ### Q1b. Normalise the attribute Input3 using the z-score transformation method.
# 
# ### Z Score = dataframe[column_name] - dataframe[column_name].mean())/dataframe[column_name].std()

# In[46]:


df['Input3'] = (df['Input3']- df['Input3'].mean())/df['Input3'].std()
print(df)


# ### Q1c.  Normalise the attribute Input12 in the range [0:0; 1:0].
# 
# ### Normed_df = (dataframe[column_name] - dataframe.min()) / (dataframe.max() - dataframe.min())

# In[48]:


max_Input12 = df['Input12'].max()
min_Input12 = df['Input12'].min()
df['Input12'] = (df['Input12'] - min_Input12) / (max_Input12 - min_Input12)
print(df)


# ### Q1d. Generate a new attribute called Average Input, which is the average of all the attributes from Input1 to Input12. This average should include the normalised attributes values but not the copies that were made of these.

# In[53]:


all_col = ['Input1','Input2','Input3','Input4','Input5','Input6','Input7','Input8','Input9','Input10','Input11','Input12']
df['Average Input']= df[all_col].mean(axis = 1)
# axis =1 denotes row wise
print(df)


# ### Q1e. Save the newly generated dataset to ./output/question1 out.csv.

# In[52]:


question1_out= df.to_csv(r'.\output\question1_out.csv',index= False,float_format = '%g')

# index is set to False (It will exclude the index column from the dataframe while saving)
# float_format = '%g' is used to avoid the round off function 


# ## Question 2: Data Reduction and Discretisation
# ### Q2a. Reduce the number of attributes using Principal Component Analysis (PCA), making sure at least 95% of all the variance is explained.
# 

# In[54]:


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# In[55]:


# Input file is read and stored in a dataframe (fileInput) 
fileInput = pd.read_csv(r".\specs\DNAData_question2.csv") 
fileInput


# In[16]:


#fit(): Training the input data. It adjusts weights according to data values so that better accuracy can be achieved.
#It internally calculates the coefficients related to pca (in this particular case) Eg: EigenVector, EigenValue
pca = PCA().fit(fileInput)


# In[57]:


# A plot of Components Vs. Variance is displayed. 
# pca.explained_variance_ratio_.cumsum() shows the Cumulative sum of the variances

plt.figure()
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.show()


# In[58]:


# PCA(n_reduced_dimensions) It is to be taken 95% atleast so the value os 0.95.
pca = PCA(0.95) 


# In[59]:


# fit_transform() method is used to transform all the coeffiecients calculated using the fit() method

dataset = pca.fit_transform(fileInput)
print(dataset)


# In[20]:


# Cumulative sum of all the variances is calculated using the cumsum()
# explained_variance_ratio returns a vector of variances for all the  dimensions that can make 95% of the varince(22 dimensions variance )
pca.explained_variance_ratio_.cumsum()


# In[62]:


# Number of dimensions used for variance of minimum 95% is returned using n_components_ property of pca object
pca.n_components_


# Thus the input dimensions gets reduced to 22 dimensions 


# ### Q2b. Discretise the PCA-generated attribute subset into 10 bins, using bins of equal width. For each component X that you discretise, generate a new column in the original dataset named pcaX width. For example, the 1st discretised principal component will correspond to a new column called pca1_width.

# In[64]:


#KBinsDiscretizer module is used to sort the input data into buckets/bins
# number_bins = 10
# encode = 'ordinal': For integer values of the bins
# strategy = 'uniform' It means the bins are of equal width 
# fit_transform: It is used to generate all the coefficients, and transform them.

from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
SameWidthArray =disc.fit_transform(dataset)
print(SameWidthArray)


# In[65]:


#Converting the np.Array to dataframe

binned_dframe= pd.DataFrame(SameWidthArray)
print(binned_dframe)


# In[66]:


# Concating the new columns(pca_ColumnNumber_Width) with the original dataset

for i in range(0,22):
    column_name = 'pca' + str(i) + '_width'
    fileInput[column_name]= binned_dframe[i]
    
print(fileInput)


# ### Q2c. Discretise PCA-generated attribute subset into 10 bins, using bins of equal frequency (they should all contain the same number of points). For each component X that you discretise, generate a new column in the original dataset named pcaX freq. For example, the 1st discretised principal component will correspond to a new column called pca1_frequency.

# In[67]:


#KBinsDiscretizer module is used to sort the input data into buckets/bins
# number_bins = 10
# encode = 'ordinal': For integer values of the bins
# strategy = 'quantile' It means the bins are of equal frequency 
# fit_transform: It is used to generate all the coefficients, and transform them.

freq_disc= KBinsDiscretizer(n_bins=10, encode='ordinal',strategy='quantile')
SamefreqArray = freq_disc.fit_transform(dataset)
print(SamefreqArray)


# In[69]:


#Converting the np.Array to dataframe

binned_freq_dframe= pd.DataFrame(SamefreqArray)
print(binned_freq_dframe)


# In[71]:


# Concating the new columns(pca_ColumnNumber_Width) with the original dataset

for i in range(0,22):
    column_name = 'pca' + str(i) + '_freq'
    fileInput[column_name]= binned_freq_dframe[i]

print(fileInput)


# ### Q2d.  Save the generated dataset

# In[73]:


#index = False is used to ignore the index in the dataframe while saving the file

fileInput.to_csv(r".\output\question2_out.csv", index =False)