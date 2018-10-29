# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:15:05 2018

@author: 311
"""


import os 
import numpy

from collections import Counter 
from sklearn.model_selection import train_test_split



def cos(X_train, X_test):#计算cos值
    testdoc_cos=[]
    test_cos=[]
    
    for row in X_test:
        vector_test = numpy.mat(row)
        for row in X_train:
            vector_train = numpy.mat(row)
            num = float(vector_test*vector_train.T)
            denom = numpy.linalg.norm(vector_test) * numpy.linalg.norm(vector_train)
            doc_cos = num / denom
        testdoc_cos.append(doc_cos)
    test_cos.append(testdoc_cos)
    
    
    return test_cos#返回一个二维的列表，每一行表示test中一篇doc和train中各篇doc的cos值


def KNN(VSMlist,lablelist,k):
    klist=[]
    classlist=[]
    kl=[]
    cl=[]
    l=[]
    test_list=[]
    
    
    X_train, X_test, Y_train, Y_test=train_test_split(VSMlist, lablelist, test_size=0.2, random_state=42)
    X_train=numpy.array(X_train)
    X_test=numpy.array(X_test)
    test_cos=cos(X_train, X_test)
    
    for each in test_cos:
        for i in range(k):
            p=each.index(max(each))
            kl.append(p)
            each.pop(p)
    klist.append(kl)
    
    for each in klist:
        for k in each:
            cl.append(Y_train[k])
        tcdict=dict(Counter(classlist).most_common(1))
        l=list(tcdict.keys())
    test_list.append(l) 

    return test_list,Y_test

def computeacc(test_list,Y_test):
    i=0
    n=len(test_list)
    acc=0
    for each in test_list:
        for c in Y_test:
           if each==c:
               i+=1
    acc=i/n
    print('The KNN result is %f'%acc )



        
        
        
        
        
        
        
    

