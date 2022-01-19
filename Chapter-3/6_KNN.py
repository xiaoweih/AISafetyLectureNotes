#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import copy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X = iris.data 
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("Training accuracy is %s"% neigh.score(X_train,y_train))
print("Test accuracy is %s"% neigh.score(X_test,y_test))


# In[2]:


import numpy as np
from math import sqrt
from collections import Counter

# Implement kNN in details
def kNNClassify(K, X_train, y_train, X_predict):
    distances = [sqrt(np.sum((x - X_predict)**2)) for x in X_train]
    sort = np.argsort(distances)
    topK = [y_train[i] for i in sort[:K]]
    votes = Counter(topK)
    y_predict = votes.most_common(1)[0][0]
    return y_predict

def kNN_predict(K, X_train, y_train, X_predict, y_predict):
    acc = 0
    for i in range(len(X_predict)):
        if y_predict[i] == kNNClassify(K, X_train, y_train, X_predict[i]):
            acc += 1
    print(acc/len(X_predict))

print("Training accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_train, y_train)
print("Test accuracy is ", end='')
kNN_predict(3, X_train, y_train, X_test, y_test)


# In[36]:


import numpy as np
import itertools
import copy 

# Attack on KNN
def kNN_attack(K, X_train, y_train, X_predict, y_predict): 
    m = np.diag([0.5,0.5,0.5,0.5])*4
    flag = True
    for i in range(1,5):
        for ii in list(itertools.combinations([0,1,2,3],i)):
            delta = np.zeros(4)
            for jj in ii:
                delta += m[jj]
                
            if y_predict != kNNClassify(K, X_train, y_train, copy.deepcopy(X_predict)+delta):
                X_predict += delta
                flag = False
                break
                
            if y_predict != kNNClassify(K, X_train, y_train, copy.deepcopy(X_predict)-delta):
                X_predict -= delta
                flag = False
                break
        if not flag:
            break
            
    print('attack data: ',X_predict)
    print('predict label: ',kNNClassify(K, X_train, y_train, X_predict))

X_test_ = X_test[0]
y_test_ = y_test[0]
print('original data: ', X_test_)
print('original label: ', y_test_)
kNN_attack(3, X_train, y_train, X_test_, y_test_)


# In[ ]:





# In[ ]:





# In[ ]:




