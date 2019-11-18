#!/usr/bin/env python
# coding: utf-8


# 
# ## Association Rules
# 
# ### Question 1: Association rules with Apriori
# 
# #### Input File: "D:\DataMining\Pracs\Practical_3\specs\gpa_question1.csv"

# In[121]:


#apriori and association_rules modules of the frequent_patterns from mlxtend are imported.

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[122]:


# Input file is loaded into a dataframe (dframe) and the column 'count 'is droped as it is not required.
dframe = pd.read_csv(r".\specs\gpa_question1.csv")
dframe = dframe.drop(['count'],axis = 1)
dframe


# In[123]:


# ML Algo sometimes cannot operate on Categorical data or they can operate on Numerical data only.
# So they are converted into numerical data using get_dummies() in pandas and saved in new dataframe.
new_df= pd.get_dummies(dframe)
new_df


# In[124]:


# Apriori algorithm is applied on new dataframe with minimum support 15%. So the itemsets with frequency >= 15% are returned.
Q1=apriori(new_df, min_support=0.15,use_colnames=True)
Q1


# In[125]:


# Result is saved in question1_out_apriori.csv
question1_out= Q1.to_csv(r'.\output\question1_out_apriori.csv',index= False)


# In[126]:


# Association rule with confidence = 90% are generated in Q1_5 using the association_rules() from mlxtend frequent_paterns
Q1_5= association_rules(Q1, metric="confidence", min_threshold=0.9)
Q1_5


# In[127]:


# Saved the association rules result in question1_out_rules9.csv
Q1_5_out= Q1_5.to_csv(r'.\output\question1_out_rules9.csv',index= False)


# In[128]:


# Association rule with confidence = 70% are generated in Q1_6 using the association_rules() from mlxtend frequent_paterns
Q1_6= association_rules(Q1, metric="confidence", min_threshold=0.7)
Q1_6


# In[129]:


# Saved the association rules result in question1_out_rules7.csv
Q1_6_out= Q1_6.to_csv(r'D:\DataMining\Pracs\Practical_3\output\question1_out_rules7.csv',index= False)


# ### Question 2 Association rules with FP-Growth
# 
# ### Input File: "D:\DataMining\Pracs\Practical_3\specs\bank_data_question2.csv"

# In[131]:


# Input file is loaded into a dataframe (dframe_Q2) and the column 'id' is droped as it is not required.
dframe_Q2 = pd.read_csv(r".\specs\bank_data_question2.csv")
dframe_Q2 = dframe_Q2.drop(['id'],axis = 1)
dframe_Q2


# In[134]:


# Column Age is binned into 3 equal width columns
age_binned = pd.cut(dframe_Q2['age'],3)
age_binned


# In[135]:


# Column Income is binned into 3 equal width columns
income_binned = pd.cut(dframe_Q2['income'],3)
income_binned


# In[136]:


# Column Children is binned into 3 equal width columns
child_binned = pd.cut(dframe_Q2['children'],3)
child_binned


# In[138]:


# All the 3 numerical data columns are filtered out and new binned columns are placed instead of original ones in the dataframe.
dframe_Q2['age'] = age_binned
dframe_Q2['income'] = income_binned
dframe_Q2['children'] = child_binned
dframe_Q2


# In[141]:


# ML Algo sometimes cannot operate on Categorical data or they can operate on Numerical data only.
# So they are converted into numerical data using get_dummies() in pandas and saved in new dataframe.
dframe_Q2_new = pd.get_dummies(dframe_Q2)
dframe_Q2_new


# In[146]:


# fpgrowth module is imported from frequenct_patterns of mlxtend.
# fpgrowth is applied on the new dataframe and frequent itemsets with minimum support 20%.
from mlxtend.frequent_patterns import fpgrowth
Q2 = fpgrowth(dframe_Q2_new, min_support=0.2, use_colnames=True)
Q2


# In[147]:


# Result is saved in question1_out_fpgrowth.csv
Q2_out= Q2.to_csv(r'.\output\question2_out_fpgrowth.csv',index= False)


# In[164]:


# A confidence value of 80% yields 8 rules. Confidence less than 80% yields a minimum of 10 rules.
Q2_5= association_rules(Q2, metric="confidence", min_threshold=0.7)
Q2_5


# In[166]:


# Association rules are saved in question2_out_rules.csv
Q2_6= Q2_5.to_csv(r'.\output\question2_out_rules.csv',index= False)

