#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import copy
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.20)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Training accuracy is %s"% gnb.score(X_train,y_train))
print("Test accuracy is %s"% gnb.score(X_test,y_test))


# In[4]:


from collections import Counter

class GaussianNB_:
    def __init__(self):
        self.prior = None
        self.avgs = None
        self.vars = None
        self.n_class = None
        
    def _get_prior(self, y):
        cnt = Counter(y)
        prior = np.array([cnt[i] / len(y) for i in range(len(cnt))])
        return prior
    
    def _get_avgs(self, X, y):
        return np.array([X[y == i].mean(axis=0) for i in range(self.n_class)])
    
    def _get_vars(self, X, y):
        return np.array([X[y == i].var(axis=0) for i in range(self.n_class)])
    
    def _get_likelihood(self, row):
        return (1 / np.sqrt(2 * np.pi * self.vars) * np.exp(-(row - self.avgs)**2 / (2 * self.vars))).prod(axis=1)
    
    def fit(self, X, y):
        self.prior = self._get_prior(y)
        self.n_class = len(self.prior)
        self.avgs = self._get_avgs(X, y)
        self.vars = self._get_vars(X, y)
        
    def predict_prob(self, X):
        likelihood = np.apply_along_axis(self._get_likelihood, axis=1, arr=X)
        probs = self.prior * likelihood
        probs_sum = probs.sum(axis=1)
        return probs / probs_sum[:, None]
    
    def predict(self, X):
        return self.predict_prob(X).argmax(axis=1)

def get_acc(y, y_hat):
    a = 0
    for i in range(len(y)):
        if y[i]==y_hat[i]:
            a += 1
    return a/len(y)


clf = GaussianNB_()
clf.fit(X_train, y_train)

y_hat = clf.predict(X_train)
acc = get_acc(y_train, y_hat)
print("Train accuracy is %s"% acc)

y_hat = clf.predict(X_test)
acc = get_acc(y_test, y_hat)
print("Test accuracy is %s"% acc)


# In[6]:


import itertools

# Attack on GaussianNB
def GaussianNB_attack(clf, X_predict, y_predict): 
    m = np.diag([0.5,0.5,0.5,0.5])*4
    flag = True
    for i in range(1,5):
        for ii in list(itertools.combinations([0,1,2,3],i)):
            delta = np.zeros(4)
            for jj in ii:
                delta += m[jj]
            
            y_pre = clf.predict(copy.deepcopy(X_predict)+delta)      
            if y_predict != y_pre:
                X_predict += delta
                flag = False
                break
                
            y_pre = clf.predict(copy.deepcopy(X_predict)-delta)      
            if y_predict != y_pre:
                X_predict -= delta
                flag = False
                break
        if not flag:
            break
    
    print('attack data: ', X_predict)
    print('predict label: ', clf.predict(copy.deepcopy(X_predict)))

X_test_ = X_test[0:1]
y_test_ = y_test[0]
print('original data: ', X_test_)
print('original label: ', y_test_)
GaussianNB_attack(clf, X_test_, y_test_)


# In[ ]:




