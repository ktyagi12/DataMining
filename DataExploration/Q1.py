#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd

workbook = pd.read_csv(r"D:\DataMining\Pracs\Prac1\specs\AutoMpg_question1.csv")


# In[34]:


missing_col= workbook.isnull().sum()
print('The missing values in the Cars data:\n', missing_col)


# In[35]:


facts=workbook.describe()

workbook['horsepower'] = workbook['horsepower'].fillna(facts.loc["mean","horsepower"])
workbook['origin'] = workbook['origin'].fillna(facts.loc["min","origin"])


workbook


# In[39]:


question1_out= workbook.to_csv(r'D:\DataMining\Pracs\Prac1\specs\output\question1_out.csv',index= False)