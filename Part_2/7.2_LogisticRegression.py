#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import copy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data 
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X[:100], y[:100], test_size=0.20)

reg = LogisticRegression(solver='lbfgs', max_iter=500)
reg.fit(X_train, y_train)
print("Training accuracy is %s"% reg.score(X_train,y_train))
print("Test accuracy is %s"% reg.score(X_test,y_test))


# In[2]:


import numpy as np
from sklearn.metrics import accuracy_score
def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

def add_b(dataMatrix):
    dataMatrix = np.column_stack((np.mat(dataMatrix),np.ones(np.shape(dataMatrix)[0])))    
    return dataMatrix

def LogisticRegression_(x_train,y_train,x_test,y_test,alpha = 0.001 ,maxCycles = 500):
    x_train = add_b(x_train)
    x_test = add_b(x_test)
    y_train = np.mat(y_train).transpose()
    y_test = np.mat(y_test).transpose()
    m,n = np.shape(x_train)     
    weights = np.ones((n,1))
    for i in range(0,maxCycles):
        h = sigmoid(x_train*weights)
        error = y_train - h
        weights = weights + alpha * x_train.transpose() * error
        
    y_pre = sigmoid(np.dot(x_train, weights))
    for i in range(len(y_pre)):        
        if y_pre[i] > 0.5:
            y_pre[i] = 1
        else:
            y_pre[i] = 0
    print("Train accuracy is %s"% (accuracy_score(y_train, y_pre)))
    
    y_pre = sigmoid(np.dot(x_test, weights))
    for i in range(len(y_pre)):        
        if y_pre[i] > 0.5:
            y_pre[i] = 1
        else:
            y_pre[i] = 0
    print("Test accuracy is %s"% (accuracy_score(y_test, y_pre)))
    
    return weights

weights = LogisticRegression_(X_train, y_train,X_test,y_test)


# In[34]:


import itertools
import copy 

# Attack on LogisticRegression
def LogisticRegression_attack(weights, X_predict, y_predict): 
    X_predict = add_b(X_predict)
    m = np.diag([0.5,0.5,0.5,0.5])*4
    flag = True
    for i in range(1,5):
        for ii in list(itertools.combinations([0,1,2,3],i)):
            delta = np.zeros(4)
            for jj in ii:
                delta += m[jj]
            delta = np.append(delta, 0.)
            
            y_pre = sigmoid(np.dot(copy.deepcopy(X_predict)+delta, weights))       
            if y_pre > 0.5:
                y_pre = 1
            else:
                y_pre = 0
            if y_predict != y_pre:
                X_predict += delta
                flag = False
                break
                
            y_pre = sigmoid(np.dot(copy.deepcopy(X_predict)-delta, weights))       
            if y_pre > 0.5:
                y_pre = 1
            else:
                y_pre = 0
            if y_predict != y_pre:
                X_predict -= delta
                flag = False
                break
        if not flag:
            break
    
    y_pre = sigmoid(np.dot(X_predict, weights))       
    if y_pre > 0.5:
        y_pre = 1
    else:
        y_pre = 0
    print('attack data: ', X_predict[0,:-1])
    print('predict label: ', y_pre)

X_test_ = X_test[0:1]
y_test_ = y_test[0]
print('original data: ', X_test_)
print('original label: ', y_test_)
LogisticRegression_attack(weights, X_test_, y_test_)


# In[ ]:




