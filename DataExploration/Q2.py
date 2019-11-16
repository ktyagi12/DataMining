#Python file for Q2

#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
df1 = pd.read_csv(r"D:\DataMining\Pracs\Prac1\specs\AutoMpg_question2_a.csv")
df2 = pd.read_csv(r"D:\DataMining\Pracs\Prac1\specs\AutoMpg_question2_b.csv")


# In[9]:


df2.rename(columns={'name': 'car_name'})


# In[12]:


df1['other'] = 1


# In[19]:


df3 = df1.append(df2,ignore_index = True, sort = True)


# In[23]:


question2_out= df3.to_csv(r'D:\DataMining\Pracs\Prac1\specs\output\question2_out.csv',index= False)


# In[24]:


print(df3)





