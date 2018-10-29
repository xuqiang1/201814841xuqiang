# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:13:55 2018

@author: 311
"""

import os 
import string
import numpy
#import random

from tkinter import _flatten             #拉平二维列表
from textblob import TextBlob            #切分
from nltk.stem import SnowballStemmer    #词干提取
from nltk.stem import WordNetLemmatizer  #词型还原
from nltk.corpus import stopwords        #停用词处理
from collections import Counter          #统计词频
from sklearn.model_selection import train_test_split   #划分train和test

#import json
#import chardet as ch #查看文件编码模块


######################################读入文本###################################
#批量打开文件，将每个文件写入列表
def getpath(mainpath):
    fp_list=[]
    lablelist=[]  # 用于生成label
    lable_class=0
   
    os.chdir(mainpath)
    fd_name=os.listdir()
    
    for each in fd_name:  
        lable_class+=1
        fd_path=mainpath+'\\'+each
        f_name=os.listdir(fd_path)
        for each in f_name:
            f_path=fd_path+'\\'+each
            fp_list.append(f_path)
            lablelist.append(lable_class)
   # print(fp_list[10])  #检查文件路径
    return fp_list,lablelist
           
def readfile(fp_list):
    filelist=[]

    for each in fp_list:
        f=open(each,"rb")
        f_read = f.read()
        #f_ch = ch.detect(f_read)  # 存储文档编码格式
        #print(f_ch)  #查看文件编码格式
        f_read_decode = f_read.decode('ISO-8859-1') # 解码文档
        filelist.append(f_read_decode)
        f.close()
    #print (filelist[0]) #查看文件
    return filelist#生成以每篇文章为元素的大列表，下面函数所用doc为其中一篇文档

'''def mark(mainpath):
    fdict_mark={}
    fp_list=[]
    os.chdir(mainpath)
    fd_name=os.listdir()
    for fd in fd_name:
        fd_path=mainpath+'\\'+fd
        f_name=os.listdir(fd_path)
        for each in f_name:     
            f_path=fd_path+'\\'+each
            fp_list.append(f_path)
        fdict_mark[fd]=fp_list[:]
        fp_list=[]
        
       
       
    print("***********测试mark**********")
    l=list(fdict_mark.keys())
    ll=list(fdict_mark.values())
    num1=len(l)
    num2=len(ll[0])
    print(l)
    print(ll[0])
    print(num1,num2)
    
    print("***********测试mark**********")
    return fdict_mark#返回为词典，词典的key是每个子文件夹的名字即分类，Value是该文件夹下文件绝对路径的列表

def randompick(fdict_mark,rate):
    tnum=0
    mun=0
    tt_filepathlist=[]
    practicedict_mark={}
    testdict_mark={}
    
    
    for key in fdict_mark.keys():
        tnum=len(fdict_mark[key])
        mun=int(rate*tnum)
        rd_filepathlist=random.choices(fdict_mark[key],k=mun)
        practicedict_mark[key]=rd_filepathlist[:]
        
        
        for each in fdict_mark[key]:
           if each not in rd_filepathlist:
               tt_filepathlist.append(each)
        testdict_mark[key]=tt_filepathlist
        tt_filepathlist=[]
    print("***********测试random**********")
    
    print(len(fdict_mark['alt.atheism']))
    print(len(practicedict_mark['alt.atheism']))
    print(len(testdict_mark['alt.atheism']))
    
    print(testdict_mark['alt.atheism'])
    
    
    
    
    print("***********测试random**********")    
    return practicedict_mark, testdict_mark#返回值为两个字典（训练和测试）组成的元组
            
def readfile(dict_mark):
    filelist=[] 
    tt=0
       
    for key in dict_mark.keys(): 
        tt+=len(dict_mark[key])
        for each in dict_mark[key]:
            f=open(each,"rb")
            f_read=f.read()
            #f_ch = ch.detect(f_read)  # 存储文档编码格式
            #print(f_ch)  #查看文件编码格式
            f_read_decode=f_read.decode('ISO-8859-1') # 解码文档
            filelist.append(f_read_decode)
            f.close()    
    print("随机选取训练的文档数量是：%d"%tt)

    #print (filelist[0]) #查看文件
    return filelist#生成以每篇文章为元素的大列表，下面函数所用doc为其中一篇文档    
''' 
    

           
    


 ######################################分词#####################################    
def cleanlines(doc):#用空格替换文档中的数字和符号
    intab= string.digits+string.punctuation #设置需要替换的数字和符号
    outtab = " "*len(string.digits+string.punctuation)#替换的空格数量==替换的数字和符号
    maketrans = str.maketrans(intab,outtab)#创建字符映射的转换表
    cl_doc = doc.translate(maketrans)
    return cl_doc


def wordtokener(doc):#切分一篇文档
    ##wordlist=[]
    docwordlist=[]
    tb_doc = TextBlob(doc)
    docwordlist=tb_doc.words
    return docwordlist

 ######################################词干提取##################################
def lemmatize(docwordlist):#词型还原
    lm_wordlist = []
    
    wnl = WordNetLemmatizer()
    for each in docwordlist:
        lm_wordlist.append(wnl.lemmatize(each))
    return lm_wordlist


def steming(docwordlist):#词干提取
    st_wordlist = []
    
    stemmer = SnowballStemmer("english")#选择一种语言
    for each in docwordlist:
        st_wordlist.append(stemmer.stem(each))
    return st_wordlist


def lowlitter(docwordlist):#大写转小写
    ll_wordlist = []
    
    for each in docwordlist:
        ll_wordlist.append(str.lower(each))
    return ll_wordlist

def dropstopwords(docwordlist):#剔除掉停用词
    dr_worklist = [w for w in docwordlist if not w in stopwords.words('english')]
    return dr_worklist
  
######################################预处理#####################################
def preprocess(filelist):#输入文档列表
    nor_wordlist = []
    
    for doc in filelist:
        doc_clean = cleanlines(doc)#替换数字和字符
        doc_tokener = wordtokener(doc_clean)#分词
        wordlist_lemmatize = lemmatize(doc_tokener)#词型还原
        ##wordlist_steming = steming(wordlist_lemmatize)#词干提取
        wordlist_lowlitter = lowlitter(wordlist_lemmatize)#大写转小写
        nor_doc_wordlist = dropstopwords(wordlist_lowlitter)#剔除停用词
        nor_wordlist.append(nor_doc_wordlist)
        #print(nor_doc_wordlist)
    #total=0
    #for each in nor_wordlist :
        #total += len(each)
    #print(total)

    print("***************************预处理结束*****************************")
    return nor_wordlist 

    
##################################计算词频,生成词典###############################

def wordfrequency(nor_wordlist,low,high): #计算文档词频并删除高频和低频词生成词典
    record=[]
    wordlist=list(_flatten(nor_wordlist))#将二维列表降为一维列表 
    frequencydict = dict(Counter(wordlist))#统计所有文档中词的词频生成字典
    #返回的是一个是字典{'a': 2, 'c': 2, 'b': 2, 'd': 1} 
    for key in frequencydict.keys():
        if frequencydict[key]<low or frequencydict[key]>high: 
            record.append(key)
    for key in record:    
        frequencydict.pop(key)

    return (frequencydict)

#####################################计算TF和IDF#################################
  
def TF(nor_doc_wordlist, frequencydict):#计算一篇文档的词频

    doc_count = {}
    df=Counter(nor_doc_wordlist)#计算一篇文章的词频生成counter类字典
    doc_Frequency=dict(df.most_common())#将counter类字典转换为字典
    for key in frequencydict.keys():
        if doc_Frequency.get(key):
            doc_count[key]=doc_Frequency[key]
        else:
            doc_count[key] = 0
 ###print('测试1111TF')
    ###for key in doc_count.keys():
       ### if doc_count[key] != 0:
          ###  print("doc_count[%s]: "%key ,doc_count[key])

    vector = numpy.array(list(doc_count.values()))
    #arlfa = 0.1
    #max_element = vector.max()
    #TF_vector = arlfa + (1-arlfa)*vector/max_element
    TF_vector = 1+numpy.log(vector)
    TF_vector = numpy.nan_to_num(TF_vector)
    
    return TF_vector 


def IDF(nor_wordlist,frequencydict):#生成IDF
    count = 0
    doc_count = {}
    N = len(nor_wordlist)
    
    for key in frequencydict.keys():
        for nor_doc_wordlist in nor_wordlist:
            if key in nor_doc_wordlist:
                count+=1
        doc_count[key]=c
    vector = numpy.array(list(doc_count.values()))
   
    IDF_vector = numpy.log((N+1)/(vector+1)+1)
    IDF_vector = numpy.nan_to_num(IDF_vector)
    IDF_vector.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\IDF.txt")
    return IDF_vector
#####################################生成VSM####################################

def VSM(nor_wordlist,frequencydict):#生成向量空间模型
    TF_vectorlist = []
    VSMlist = []

    IDF_vector = IDF(nor_wordlist,frequencydict)

    for nor_doc_wordlist in nor_wordlist:
        TF_vector = TF(nor_doc_wordlist,frequencydict)
        TF_vectorlist.append(TF_vector)
        VSM_vector = TF_vector*IDF_vector
        #print(IDF_vector,TF_vector,VSM_vector)
        #print(VSM)
        VSMlist.append(VSM_vector)

    #保存所有文档的TF
    TF_array = numpy.array(TF_vectorlist)
    print("the TF_array is ",TF_array.shape)
    TF_array.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\TF.txt")
    #保存所有文档的向量列表
    VSM_array = numpy.array(VSMlist)
    print('the VSM_array is ',VSM_array.shape)
    VSM_array.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\VSM.txt")

    return VSMlist

    
#####################################KNN####################################           
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
    kl=[]
    cl=[]
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
        tcdict=dict(Counter(cl).most_common(1))
        test_list.append(tcdict.keys())

    print(test_list)
    print(len(test_list))
    print('**********测试**********')
    print(Y_test)
    print(len(Y_test))
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
    
#####################################main#################################### 
   

mainpath="C:\\Users\\311\\Desktop\\data mining\\20news-18828" 
fp_list=getpath(mainpath)[0]
lablelist=getpath(mainpath)[1]

filelist=readfile(fp_list)
nor_wordlist=preprocess(filelist)
frequencydict=wordfrequency(nor_wordlist,16,1500) 
c=len(frequencydict)
print("词典数量%d"%c)
VSMlist=VSM(nor_wordlist,frequencydict)
k=3
knn=KNN(VSMlist,lablelist,k)
test_list=knn[0]
Y_test=knn[1]
computeacc(test_list,Y_test)
       
   