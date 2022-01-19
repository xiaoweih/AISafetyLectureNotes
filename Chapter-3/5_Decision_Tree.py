#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import copy

# Get iris
def get_iris():
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data 
    y = iris.target

    data_iris = []
    for i in range(len(X)):
        dict = {}
        dict['f0'] = X[i][0]
        dict['f1'] = X[i][1]
        dict['f2'] = X[i][2]
        dict['f3'] = X[i][3]

        dict['label'] = y[i]
        data_iris.append(dict)
    return data_iris
    
data = get_iris()
label = 'label'


def entropy(data, label):
    cl = {}
    for x in data:
        if x[label] in cl:
            cl[x[label]] += 1
        else:
            cl[x[label]] = 1
    tot_cnt = sum(cl.values())
    return sum([ -1 * (float(cl[x])/tot_cnt) * math.log2(float(cl[x])/tot_cnt) for x in cl])


def findInformationGain(data, label, column, entropyParent):
    keys = { i[column] for i in data }
    entropies = {}
    count = {}
    avgEntropy = 0
    for val in keys:
        modData = [ x for x in data if x[column] == val]
        entropies[val] = entropy(modData, label)
        count[val] = len(modData)
        avgEntropy += (entropies[val] * count[val])

    tot_cnt = sum(count.values())
    avgEntropy /= tot_cnt
    return entropyParent - avgEntropy

node = 0
nodeMapping = {}
edges = []

def makeDecisionTree(data, label, parent=-1, branch=''):

    global node, nodeMapping
    if parent >= 0:
        edges.append((parent, node, branch))

    #find the variable(column) with maximum information gain
    infoGain = []
    columns = [x for x in data[0]]
    for column in columns:
        if not(column == label):
            ent = entropy(data, label)
            infoGain.append((findInformationGain(data, label, column, ent), column))
    splitColumn = max(infoGain)[1]

    # Leaf node, final result, if maximum information gain is not significant
    if max(infoGain)[0] < 0.01:
        nodeMapping[node] = data[0][label]
        node += 1
        return
    nodeMapping[node] = splitColumn
    parent = node
    node += 1
    branchs = { i[splitColumn] for i in data }#All out-going edges from current node
    for branch in branchs:

        # Create sub table under the current decision branch
        modData = [x for x in data if splitColumn in x and x[splitColumn] == branch]
        for y in modData:
            if splitColumn in y:
                del y[splitColumn]

        # Create sub-tree
        makeDecisionTree(modData, label, parent, branch)

makeDecisionTree(data, label)

print('nodemapping ==> ', nodeMapping, '\n\nedges ===>', edges)


path = []
label_x = None

# QUERY
def query(i, data_x):
    global path, label_x
    path.append(i)
    next_q = False
    for e in edges:
        if e[0]==i:
            next_q=True
            break
        
    if next_q:
        for e in edges:
            if e[0]==i and e[2]==data_x[str(nodeMapping[i])]:
                i = e[1]
                query(i, data_x)
                break
    else:
        label_x = nodeMapping[i]
        
data_x = get_iris()[149]
query(0, data_x)
print()
print('original_data:', data_x)
print('original_path:',path,' predict_label:', label_x)

# ATTACK
attack_label = None
attack_path = None

def judge_e(i):
    next_ = False
    for e in edges:
        if e[0]==i:
            next_=True
            break
    return next_
    
def atk_path(path_,i):
    global attack_label, attack_path
    for e in edges:
        ppath = copy.deepcopy(path_)
        if e[0]==i:
            ppath.append(e[1])
            if judge_e(e[1]):
                atk_path(ppath,e[1])
            elif nodeMapping[e[1]]!=label_x and attack_label==None:
                attack_path = ppath
                attack_label = nodeMapping[e[1]]
            
def attack():
    for i in range(1,len(path)):
        atk_path(path[:-i],path[-1-i])
        if attack_label != None:
            break
            
attack()
print('attack_path:',attack_path,' attack_label:', attack_label)


# In[ ]:




