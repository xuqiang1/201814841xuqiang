# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:13:55 2018

@author: 311
"""

import os 
import string
import numpy
import datetime
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
    print('读取的文档数为%d'%len( filelist))
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    print("***************************文件读取完成*****************************")
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
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
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
        
        
    c=len(frequencydict)
    print("生成的词典数量为%d"%c)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    print("***************************词典生成完毕*****************************")
    return (frequencydict)

#####################################计算TF和IDF#################################
  
def TF(nor_doc_wordlist, frequencydict):#计算一篇文档的词频

    TFdoc_count = {}
    df=Counter(nor_doc_wordlist)#计算一篇文章的词频生成counter类字典
    doc_Frequency=dict(df.most_common())#将counter类字典转换为字典
    for key in frequencydict.keys():
        if doc_Frequency.get(key):
            #TFdoc_count[key]=doc_Frequency[key]
            TFdoc_count.setdefault(key,doc_Frequency[key])
        else:
            TFdoc_count.setdefault(key,0)
 ###print('测试1111TF')
    ###for key in doc_count.keys():
       ### if doc_count[key] != 0:
          ###  print("doc_count[%s]: "%key ,doc_count[key])

    vector = numpy.array(list(TFdoc_count.values()))
    #a= 0.1
    #max_element = vector.max()
    #TF_vector = a + (1-a)*vector/max_element
    #print('TF_vector.max is')
    #print(TF_vector.max())
    TF_vector = 1+numpy.log(vector)
    TF_vector = numpy.nan_to_num(TF_vector)
    
    return TF_vector 


def IDF(nor_wordlist,frequencydict):#生成IDF
    count = 0
    IDFdoc_count = {}
    N = len(nor_wordlist)
    
    for key in frequencydict.keys():
        for nor_doc_wordlist in nor_wordlist:
            if key in nor_doc_wordlist:
                count+=1

        IDFdoc_count.setdefault(key,count)
        count=0
    
    vector = numpy.array(list(IDFdoc_count.values()))
   
    
    IDF_vector = 1+numpy.log((N+1)/(vector+1))  
    #IDF_vector = numpy.log(N/vector)
    IDF_vector = numpy.nan_to_num(IDF_vector)
    print("the IDF_array is ",IDF_vector.shape)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    #print(IDF_vector)
    #IDF_vector.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\IDF.txt")
    
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
    #print(TF_array)
    #TF_array.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\TF.txt")
    #保存所有文档的向量列表
    VSM_array = numpy.array(VSMlist)
    print('the VSM_array is ',VSM_array.shape)
    #print(VSM_array)
    #VSM_array.tofile(r"C:\Users\311\Desktop\data mining\201814841xuqiang\homework01\output\VSM.txt")
    
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    
    print("***************************VSM生成完毕*****************************")
    return VSMlist

    
#####################################KNN####################################           


def cos22(X_train, X_test):#计算cos函数
    testdoc_cos=[]
    test_cos=[]
    len_test=0.0
    len_train=0.0
    num=0.0
    
    for c in range(len(X_test)):
        len_test=numpy.linalg.norm(X_test[c])
        #for i in X_test[c]:
            #len_test+=(i**2)
            #print('len_test是%f'%len_test)
        for vector_train in X_train:
            #for j in vector_train:
                #len_train+=(j**2)
        #print('len_train是%f'%len_train) 
        
            len_train=numpy.linalg.norm(vector_train)
            num=numpy.dot(X_test[c], vector_train)
            if len_test!=0 and len_train!=0:
                cos=num/((len_test*len_train)**0.5)
            else:
                cos=0
        
            testdoc_cos.append(cos)
            len_train=0
        test_cos.append(testdoc_cos)
        print(c)
        testdoc_cos=[]
        len_test=0
        #print(testdoc_cos)
    test_cos_array=numpy.array(test_cos)
    print("the test_cos_array is ",test_cos_array.shape)
    


    return test_cos#返回一个二维的列表，每一行表示test中一篇doc和train中各篇doc的cos值    



def count_cosinvalue(x_train, x_test):
  
    
    inner_product = numpy.dot(x_test, x_train.T)
    norm_train = numpy.linalg.norm(x_train, ord=2, axis=1, keepdims=False)
    norm_test = numpy.linalg.norm(x_test, ord=2, axis=1, keepdims=False)
    norm_nd = numpy.dot(numpy.array([norm_test]).T, numpy.array([norm_train]))
    cosin_values = inner_product / norm_nd
    
    test_cos=numpy.nan_to_num(cosin_values)
    test_cos.tolist() 

    test_cos_array=numpy.array(test_cos)
    print("the test_cos_array is ",test_cos_array.shape)
    
    return test_cos










def KNN(VSMlist,lablelist,k):
    klist=[]
    kl=[]
    cl=[]
    t_list=[]
    test_list=[]
    
    
    X_train, X_test, Y_train, Y_test=train_test_split(VSMlist, lablelist, test_size=0.2, random_state=42)
    print('X_train文档数是%d'%len(X_train))
    print('X_test文档数是%d'%len(X_test))
    print('Y_train文档数是%d'%len(Y_train))
    print('Y_test文档数是%d'%len(Y_test))
    X_train=numpy.array(X_train)
    X_test=numpy.array(X_test)
    test_cos=cos22(X_train, X_test)
    
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    print("***************************COS计算完毕*****************************")  
    

    
    for each in test_cos:
        row=each.tolist() 
        
        for i in range(k):
            p=row.index(max(row))
            kl.append(p)
            row[p]=0
        klist.append(kl)
        kl=[]
        #print(kl)
              
    #print('最大的k个位置为')
    #print(klist[0])
    #print(len(klist[0]))
    
    for each in klist:
        for k in each:
            cl.append(Y_train[k])
            #print(Y_train[k])
        #print('#########chehsi#####')    
        #print(Y_train[0],Y_train[1],Y_train[2])
        #print(cl)
        tcdict=dict(Counter(cl).most_common(1))
        cl=[]
        #print('最大词典')
        #print(tcdict)
        
        t_list.append(list(tcdict.keys()))
        
        test_list=list(_flatten(t_list))

    #print(test_list)
    #print(Y_test)
    
    print('**********测试**********')
    #print(len(test_list))
    #print(len(Y_test))
    print("***************************KNN*****************************")
    return test_list,Y_test

def computeacc(test_list,Y_test):
    i=0
    j=0
    n=len(test_list)
    acc=0
    for k in range(n):
        j+=1
        if test_list[k]==Y_test[k]:
            i+=1
        #print('%d in %d'%(i,j))
    acc=i/n
    print('The KNN result is %f'%acc ) 
    return acc
    
    

    
#####################################main#################################### 
   
  

mainpath="C:\\Users\\311\\Desktop\\data mining\\20news-18828" 
nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print(nowTime,'\n')
gp=getpath(mainpath)
fp_list=gp[0]
lablelist=gp[1]

filelist=readfile(fp_list)
nor_wordlist=preprocess(filelist)
frequencydict=wordfrequency(nor_wordlist,12,1500) 

VSMlist=VSM(nor_wordlist,frequencydict)

knn=KNN(VSMlist,lablelist,5)
test_list=knn[0]
Y_test=knn[1]   
    

nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
print(nowTime,'\n')

#print(acclist)
       




   