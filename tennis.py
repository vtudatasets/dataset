
# coding: utf-8

# # ID3 Algorithm

# In[1]:


import numpy as np
import pandas as pd
import pprint


# In[2]:


tennis = pd.read_csv('C:/Users/Goutham/Desktop/tennis.csv')


# In[ ]:


tennis
#type(tennis)


# In[5]:


#data = golf
#output_col = 'Play'
#col_name = 'Outlook'

def get_uniq_values(data, col_name) :
    vals = data[col_name].unique()
    return(vals)


# In[6]:


def calculate_entropy (classes) :
    probs = classes.value_counts() / len(classes)
    ent = (-probs * np.log2(probs)).sum()
    return(ent)


# In[7]:


def calculate_entropy_branch (data, colname, outputcol) :
    groups = get_uniq_values(data, colname)
    
    entropy = 0
    
    for group in groups : 
        labels = data[outputcol][data[colname] == group]
        branch_prob = len(labels) / len(data)
        branch_entropy = calculate_entropy(labels)
        entropy += branch_prob * branch_entropy
    
    return(entropy)


# In[8]:


def calculate_information_gain (data, colname, outputcol) :
    root_entropy = calculate_entropy(data[outputcol])
    branch_entropy = calculate_entropy_branch(data, colname, outputcol)
    ig = root_entropy - branch_entropy
    return (ig)


# In[9]:


def calculate_best_feature (data, features, outputcol) : 
    #features = data.columns.difference([outputcol])
    
    res = {}
    
    for feature in features : 
        res[feature] = calculate_information_gain(data, feature, outputcol)
        #print(res)
    
    return(res)


# In[10]:


def ID3(data, originaldata, features, outputcol, parentclass):
   
    if len(data[outputcol].unique()) <= 1 : 
        return(data[outputcol].unique())
    elif len(data) == 0 : 
        return (originaldata[outputcol].mode())
    elif len(features) == 0 :
        return (parentclass)
    else :
        parentclass = data[outputcol].mode()
        
        item_values = calculate_best_feature(data, features, outputcol)
        best_feature = max(item_values, key = item_values.get)
        features.remove(best_feature)
        
        tree = {best_feature:{}}
                        
        for value in data[best_feature].unique() :
            value = value
            
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            # sub_data = data.where(data[best_feature] == value).dropna()
            sub_data = data[data[best_feature] == value]
            #print(sub_data)
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, data, features, outputcol, parentclass)
            #print(subtree)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
            
        return(tree)   


# In[11]:


def predict(query, tree, default = 'yes'):
    
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            result = tree[key][query[key]]

            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result
        


# In[ ]:


tree = ID3(tennis, tennis, ['Outlook', 'Temp', 'Humidity', 'Wind'], 'Play', 'yes')
pprint.pprint(tree)


# In[ ]:


query = {'Outlook' : 'Sunny', 'Temperature' : 'Hot', 'Humidity' : 'High'}

predict(query, tree)

