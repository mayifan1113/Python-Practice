
# coding: utf-8

# In[3]:

get_ipython().system(' pip install -U scikit-learn')


# In[2]:

get_ipython().system(' python -m pip install --upgrade pip')


# In[39]:

import numpy as np
#this is the example about similarity
dataset_filename = 'E:/My Learning/Learning data mining/LearningDataMiningwithPythonSecondEdition_Code/Chapter01/affinity_dataset.txt'
X = np.loadtxt(dataset_filename)


# In[40]:

print(X[:5])


# In[41]:

X[0:3]


# In[42]:

#count the # of customers who bought apples
number_apple = 0
for sample_apple in X:
    if sample_apple[3] == 1:
        number_apple += 1
print(str(number_apple) + ' people bought apples.')


# In[43]:

#generate a default dictionary
from collections import defaultdict
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)


# In[44]:

#get the number of rows and columns
n_samples, n_features = X.shape
n_samples, n_features


# In[46]:

#check if sample customer bought product premise
for sample in X:
    for premise in range(5):
#if customer didn't buy this premise, skip to the next
        if sample[premise] == 0:
            continue
#if customer bought product premise, then add 1 occurance to product premise    
        else:
            num_occurances[premise] += 1
#exclude all the cases when product premise and product conclusion are the same
        for conclusion in range(n_features):
            if premise == conclusion:
                continue
            if sample[conclusion] == 1:
                valid_rules[(premise, conclusion)] += 1
            else:
                invalid_rules[(premise,conclusion)] += 1


# In[48]:

#calculate the value of support and confidence
#support is based on number of cases matched valid rule
support = valid_rules

#confidence is based on ratio of valid number against number of occurances
confidence = defaultdict(float)
for premise, conclusion in valid_rules.keys():
    confidence[(premise,conclusion)] = valid_rules[(premise, conclusion)] / num_occurances[premise]


# In[49]:

#define a function
#input: premise, conclusion, support dictionary, confidence dictionary, features
#output: support and confidence of each rule (the combination of premise and conclusion)
def rule_of_feature (premise, conclusion, support, confidence, features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
#change the index of a feature to a feature name
    print('if people buy {0} they will also buy {1}'.format(premise_name, conclusion_name))
    print(' - Support: {0}'.format(support[(premise,conclusion)]))
    print(' - Confidence: {0:.3f}'.format(confidence[(premise,conclusion)]))


# In[66]:

features = ['bread','milk','cheese','apple','banana']
premise = 2
conclusion = 3
rule_of_feature(premise,conclusion,support,confidence,features)


# In[68]:

from operator import itemgetter
#display all the elements in the support dictionary
support.items()


# In[74]:

#sort the list by descending support 
sorted_support = sorted(support.items(), key = itemgetter(1), reverse = True)
#display top 5 supports
sorted_support[0:5]


# In[76]:

#display the top 5 support combinations of premise and conclusion
for id in range(5):
    print('Conclusion #{0}:'.format(id + 1))
    premise, conclusion = sorted_support[id][0]
    rule_of_feature(premise, conclusion, support, confidence,features)


# In[92]:

#sort the list by descending confidence
sorted_confidence = sorted(confidence.items(), key = itemgetter(1), reverse = True)
#display top 5 confidence
sorted_confidence[0:5]


# In[79]:

#display the top 5 conclusion combinations of premise and conclusion
for id in range(5):
    print('Conclusion #{0}:'.format(id + 1))
    premise, conclusion = sorted_confidence[id][0]
    rule_of_feature(premise, conclusion, support, confidence,features)


# In[90]:

#classification example ---- iris (inside the scikit-learn database)
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target


# In[91]:

#check the description of the iris dataset
print(dataset.DESCR)


# In[ ]:



