
# coding: utf-8

# # Menu Engineering

# ## Files used in this project
# #### 1. Different locations file
# #### 2. different menu purchase file
# #### 3. menu item details file
# 
# ## xxx is used to coverup real attributes and file names 

# # Import Libraries

# In[1]:

get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc


# # Import Api's

# In[2]:

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# # read Locations table

# In[3]:

reader = pd.read_table('xxx.csv', chunksize=50000,low_memory=False,sep=",",usecols=[14,15,16,9,4])


# In[4]:

def fil():
    full_menu = pd.DataFrame([]) 
    for chunk in reader:
        x = chunk[(chunk["xxx"]==1) & (chunk["xxx"]=="Menu Item")]
        full_menu = full_menu.append(x)
    return full_menu


# In[5]:

ful = fil()


# In[6]:

ful


# # read menu item table

# In[7]:

menu_item = pd.read_csv('xxxx.csv',usecols=['menuxxx','menuxxxxx'])


# In[8]:

menu_item.head()


# In[9]:

menu_item_clus = pd.read_csv('xxxxxxx.csv')


# In[10]:

menu_item_clus.head(10)


# In[11]:

menu = pd.merge(menu_item,menu_item_clus, how = 'left', left_on = 'menuxxx', right_on = 'menuxxx')[['menuItemxx',  'xxID','xxxMaster']] 


# In[12]:

menu.head(100)


# # join menu item and location sites

# In[13]:

df = pd.merge(ful,menu_item, how = 'left', left_on = 'xxxID', right_on = 'xxxID')[['xxxID','menuxxx','menuxxxx1','xxx']] 


# In[14]:

df = pd.merge(df,menu_item_clus, how = 'left', left_on = 'xxxID', right_on = 'xxx')[['xxx',  'xxx','xxx','xxx']] 


# In[15]:

df


# In[16]:

df['xxxID'].nunique()


# In[17]:

df['xxxxID'].unique()


# In[18]:

df = df.groupby(['xxxID','xxxID','xxx','xxx'])['xxxx'].count().reset_index(name='Quantity')


# In[19]:

basket = (df.groupby(['xxx', 'xxx'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('xxxID'))
basket.head()


# In[20]:

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# In[21]:

basket_sets = basket.applymap(encode_units)
basket_sets.head()


# In[22]:

frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
frequent_itemsets


# In[23]:

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)


# In[24]:

final_rules = rules[(rules['support'] >= 0.1) & (rules['confidence'] >= 0.1)]


# In[25]:

final_rules


# # Frequent item sets per location

# In[26]:

basket2 = (df[df['Location'] == 'xxxx ST'].groupby(['xxxID', 'xxxx'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('xxxID'))
basket_sets2 = basket2.applymap(encode_units)
basket_sets2





# In[27]:

frequent_itemsets2 = apriori(basket_sets2, min_support=0.01, use_colnames = True)
frequent_itemsets2


# In[28]:

rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold = 0)
rules2


# In[29]:

rules2[(rules2['support'] >= 0.05) & (rules2['confidence'] >= 0.3)]


# # visualization

# In[30]:

dfp = df.groupby(['xxxxID'])['xxxxID'].count().reset_index(name='no_orders')


# In[31]:

bins= [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
plt.hist(dfp['no_orders'].values, bins=bins, edgecolor="k")
plt.xticks(bins)
plt.show()


# In[33]:

## Average of menu items per order


# In[34]:

dfp["no_orders"].mean()


# In[ ]:



