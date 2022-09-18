#!/usr/bin/env python
# coding: utf-8

# In[2]:
# import os

from sklearn import datasets
dataset = datasets.load_iris()
#dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

observations = len(X)
features = len(dataset.feature_names)
classes = len(dataset.target_names)
print("Number of Observations: " + str(observations))
print("Number of Features: " + str(features))
print("Number of Classes: " + str(classes))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[3]:


############## decision tree

print("===== decision tree ======")

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print("Training accuracy is %s"% tree.score(X_train,y_train))
print("Test accuracy is %s"% tree.score(X_test,y_test))

print("Labels of all instances:\n%s"%y_test)
y_pred = tree.predict(X_test)
print("Predictive outputs of all instances:\n%s"%y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"%confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s"%classification_report(y_test, y_pred))


# In[4]:


############## k-nn

print("===== k-nn ======")

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
print("Training accuracy is %s"% neigh.score(X_train,y_train))
print("Test accuracy is %s"% neigh.score(X_test,y_test))

print("Labels of all instances:\n%s"%y_test)
y_pred = neigh.predict(X_test)
print("Predictive outputs of all instances:\n%s"%y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"%confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s"%classification_report(y_test, y_pred))


# In[5]:


############## Logistic Regression 

print("===== Logistic Regression ======")

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression(solver='lbfgs', max_iter=10000)
reg.fit(X_train, y_train)
print("Training accuracy is %s"% reg.score(X_train,y_train))
print("Test accuracy is %s"% reg.score(X_test,y_test))

print("Labels of all instances:\n%s"%y_test)
y_pred = reg.predict(X_test)
print("Predictive outputs of all instances:\n%s"%y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"%confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s"%classification_report(y_test, y_pred))


# In[6]:


############## Naive Bayes

print("===== Naive Bayes ======")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("Training accuracy is %s"% gnb.score(X_train,y_train))
print("Test accuracy is %s"% gnb.score(X_test,y_test))

print("Labels of all instances:\n%s"%y_test)
y_pred = gnb.predict(X_test)
print("Predictive outputs of all instances:\n%s"%y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n%s"%confusion_matrix(y_test, y_pred))
print("Classification Report:\n%s"%classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




