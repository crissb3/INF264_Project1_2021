import numpy as np
import pandas as pd
import math
import json
from collections import Counter
from sklearn.model_selection import train_test_split
header = ["Outlook", "Temperature", "Humidity", "Wind", "Label"]
#header = ["Feature1", "Feature2","Feature3","Feature4","Feature5","Feature6","Feature7","Feature8","Feature9","Feature10", "Label"]
#data_in = pd.read_csv("magic04.data", names=header)
data_without_headers = pd.read_csv("tennis.data")
nrFeatures = len(data_without_headers.columns)
headertest = []
for i in range(1,nrFeatures):
    headertest.append("Feature "+str(i))
headertest.append('Label')
data_with_headers = pd.read_csv("tennis.data", names=header)


#print((data_in).head())

# Feature values X
X = data_with_headers #.iloc[:, :4]
# Label values Y
y = data_with_headers.iloc[:, -1]






# Entropy method. This function takes a column of the dataset, and calculates its entropy
def entropy(dataset_column):
    values, counter = np.unique(dataset_column, return_counts=True)
    entropy_val = 0
    for i in range(len(values)):
        entropy_val -= ((counter[i] / sum(counter)) * np.log2(counter[i] / (sum(counter))))

    return abs(entropy_val)


def entropy_of_attributes(dataset, feature, feature_value):
    opt1 = 0
    opt2 = 0
    decision = dataset.iloc[:,-1]
    decision_options = list(set(decision))
    #print(decision_options)
    for i, j in zip(dataset[feature], dataset.iloc[:,-1]):
        if i == feature_value and j == decision_options[0]:
            opt1 += 1
        if i == feature_value and j == decision_options[1]:
            opt2 += 1
    #print(opt1, opt2)
    if opt1 == 0 or opt2 == 0:
        return 0
    else:
        sum = opt1 + opt2
        entropy = 0 - (((opt1/sum)*math.log2(opt1/sum)) + ((opt2/sum)*math.log2(opt2/sum)))
        return entropy


def conditional_entropy(dataset, feature):
    cond_ent = 0
    diff_feature_values = list(set(dataset[feature]))
    feature_value = list(dataset[feature])
    counts_of_feature_values = Counter(feature_value)

    for i in diff_feature_values:

        cond_ent = cond_ent + (entropy_of_attributes(dataset,feature,i) * counts_of_feature_values.get(i) / len(dataset[feature]))

    return cond_ent

def information_gain(dataset,feature):
    ig = entropy(dataset.iloc[:,-1]) - conditional_entropy(dataset,feature)
    #print(ig)
    return ig

def highest_info_gain(dataset):
   # print("YOYOO",len(dataset.columns), dataset.columns)
    size = len(dataset.columns) - 1
    best = 0
    best_index = 0
    features = list(dataset.columns)
    for i in range(size):
        feature = features[i]
        #print("AAAAAAAAAAAAAAAAAAA",feature)
        ig = information_gain(dataset, feature)

        if best < ig:
            best = ig
            best_index = i
    return best_index, features[best_index]

def learn(X, y, impurity_measure='entropy', tree = None):
    label = X.keys()[-1]

    nodeIndex, nodeName = highest_info_gain(X)

    diff_attributes = list(set(X[nodeName]))
    attributes = list(X[nodeName])
    counts_of_attributes = Counter(attributes)

    if tree == None:
        tree = {}
        tree[nodeName] = {}
    print(tree)
    for attribute in diff_attributes:

        test = X[X[nodeName] == attribute].reset_index(drop=True)
        print(test)
        testValue, testCounts = np.unique(test.iloc[:,-1], return_counts=True)
        print(testValue,testCounts)

        if len(testCounts) == 1:
            tree[nodeName][attribute] = testValue[0]
        else:
            tree[nodeName][attribute] = learn(test,y=test.iloc[:,-1])

    return tree



print(entropy(X['Label']))
#learn(X,y)
#print(entropy_of_attributes(X,'Wind','Strong'))
#print("COND ent",conditional_entropy(X,'Wind'))

#igtest = information_gain(X,'Wind')
#print(igtest)

best_index, featurename_of_best_index = highest_info_gain(X)
print("Index of feature with highest information gain:",best_index, "Feature name:", featurename_of_best_index)

def showTree(tree):
    print(json.dumps(tree, sort_keys=True, indent = 5))

test = learn(X,y)
showTree(test)
