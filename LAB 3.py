#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)
def entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy
class_entropy = entropy(df['buys_computer'])
features = df.columns[:-2] 
informationgains = {}

for feature in features:
    weighted_entropy = 0
    for value in df[feature].unique():
        subset = df[df[feature] == value]
        subset_entropy = entropy(subset['buys_computer'])
        weight = len(subset) / len(df)
        weighted_entropy += weight * subset_entropy
    informationgain = class_entropy - weighted_entropy
    informationgains[feature] = informationgain

root = max(informationgains, key=informationgains.get)

print("Information Gains:")
for feature, gain in informationgains.items():
    print(f"{feature}: {gain}")

print(f"first feature for constructing the decision tree is: {root}")


# In[5]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buyscomputer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
df2= pd.get_dummies(df, columns=['age', 'income', 'student', 'credit_rating'])
X = df2.drop('buyscomputer', axis=1)
y = df2['buyscomputer']
model = DecisionTreeClassifier()
model.fit(X, y)
tree_depth = model.get_depth()
print("Tree Depth:", tree_depth)



# In[ ]:




