# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:13:55 2018

@author: 311
"""
import os 
import string
import numpy
import datetime
import random

from tkinter import _flatten             #拉平二维列表
from textblob import TextBlob            #切分
from nltk.stem import SnowballStemmer    #词干提取
from nltk.stem import WordNetLemmatizer  #词型还原
from nltk.corpus import stopwords        #停用词处理
from collections import Counter          #统计词频
from sklearn.model_selection import train_test_split   #划分train和test
#import chardet as ch #查看文件编码模块

######################################读入文本划分训练和测试数据集###################################

def getpath(mainpath,rate):
    """
    获取每个文档绝对路径
    :param :mainpath,rate : 文档主路径，测试数据比例
    :return :trainpath_dict,testpath_dict,traindocnum : 返回值为两个字典（训练和测试）和训练集doc数量组成的元组 
    """
    fp_list=[]
    trainpath_dict={}
    testpath_dict={}
    lable_class=0
    path_dict={}
    l1=0
    l2=0
    
    os.chdir(mainpath)
    fd_name=os.listdir()   
    for each in fd_name:  
        fd_path=mainpath+'\\'+each
        f_name=os.listdir(fd_path)
        for each in f_name:
            f_path=fd_path+'\\'+each
            fp_list.append(f_path)
        path_dict.setdefault(lable_class,fp_list)#文档路径字典，键为文档分类（0-19），值为属于该类文档的绝对路径的列表
        fp_list=[]
        lable_class+=1   
     
    for key in path_dict.keys():#划分训练和测试数据
       r=random.randint(1,99)
       trainpath_list,testpath_list=train_test_split(path_dict[key], test_size=rate, random_state=r)
       trainpath_dict.setdefault(key,trainpath_list)
       testpath_dict.setdefault(key,testpath_list)
       l1+=len(trainpath_list)
       l2+=len(testpath_list)
    traindocnum=l1
    print('*********测试********') 
    print('训练集的文档数量：%d'%l1)       
    print('测试集的文档数量：%d'%l2) 
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    return trainpath_dict,testpath_dict,traindocnum#返回值为两个字典（训练和测试）组成的元组      
         
def readfile(path_dict):
    """
    读出文档并储存在字典值为嵌套列表的结构中
    :param  :path_dict : trainpath_dict或testpath_dict
    :return :file_dict : 字典值为以每篇文章为元素的列表
    """
    file_list=[]
    file_dict={}
    for key in path_dict.keys():   
        for each in path_dict[key]:
            f=open(each,"rb")
            f_read = f.read()
            #f_ch = ch.detect(f_read)  # 存储文档编码格式
            #print(f_ch)  #查看文件编码格式
            f_read_decode = f_read.decode('ISO-8859-1') # 解码文档
            file_list.append(f_read_decode)
            f.close()
        file_dict.setdefault(key,file_list)
        file_list=[]        
    #print (filelist[0]) #查看文件
    print("***************************文件读取完成*****************************")
    return file_dict

 ######################################分词#####################################    
def cleanlines(doc):
    """
    用空格替换文档中的数字和符号
    :param  :doc:filelist嵌套列表中的一个元素即一篇文档
    :return :cl_doc：一篇文档组成的列表
    """
    intab= string.digits+string.punctuation #设置需要替换的数字和符号
    outtab = " "*len(string.digits+string.punctuation)#替换的空格数量==替换的数字和符号
    maketrans = str.maketrans(intab,outtab)#创建字符映射的转换表
    cl_doc = doc.translate(maketrans)
    return cl_doc

def wordtokener(doc):
    """
    切分一篇文档
    :param doc : filelist嵌套列表中的一个元素即一篇文档
    :return : docwordlist：一篇文档组成的列表
    """
    docwordlist=[]
    tb_doc = TextBlob(doc)
    docwordlist=tb_doc.words
    return docwordlist

 ######################################词干提取##################################
def lemmatize(docwordlist):
    """
    词型还原
    :param  :docwordlist : wordtokener()函数切分好的的由一篇文档单词组成的列表
    :return :lm_wordlist : 处理后由一篇文档单词组成的列表
    """
    lm_wordlist = []
    wnl = WordNetLemmatizer()
    for each in docwordlist:
        lm_wordlist.append(wnl.lemmatize(each))
    return lm_wordlist

def steming(docwordlist):
    """
    词干提取
    :param  :docwordlist : wordtokener()函数切分好的的由一篇文档单词组成的列表
    :return :st_wordlist : 处理后由一篇文档单词组成的列表
    """
    st_wordlist = []
    stemmer = SnowballStemmer("english")#选择一种语言
    for each in docwordlist:
        st_wordlist.append(stemmer.stem(each))
    return st_wordlist

def lowlitter(docwordlist):
    """
    大写转小写
    :param  :docwordlist : wordtokener()函数切分好的的由一篇文档单词组成的列表
    :return :ll_wordlist : 处理后由一篇文档单词组成的列表
    """
    ll_wordlist = []
    for each in docwordlist:
        ll_wordlist.append(str.lower(each))
    return ll_wordlist

def dropstopwords(docwordlist):
    """
    剔除掉停用词
    :param  :docwordlist : wordtokener()函数切分好的的由一篇文档单词组成的列表
    :return :ll_wordlist : 处理后由一篇文档单词组成的列表
    """
    dr_worklist = [w for w in docwordlist if w not in stopwords.words('english') and 3<len(w)]
    #print('stopwords处理前文章前单词数%d,处理后单词数%d'%(len(docwordlist),len(dr_worklist)))
    return dr_worklist
  
######################################预处理#####################################
def preprocess(file_dict):
    """
    调用上述函数进行预处理
    :param  :filelist : 字典值为以每篇文章为元素的列表
    :return :file_worddict : 字典值为处理过的每篇文章单词为元素的嵌套列表
    """
    file_worddict = {}
    
    for key in file_dict.keys():
        file_wordlist = []
        for doc in file_dict[key]:
            doc_clean = cleanlines(doc)#替换数字和字符
            doc_tokener = wordtokener(doc_clean)#分词
            wordlist_lemmatize = lemmatize(doc_tokener)#词型还原
            wordlist_lowlitter = lowlitter(wordlist_lemmatize)#大写转小写
            wordlist_steming = steming(wordlist_lowlitter)#词干提取
            doc_wordlist = dropstopwords(wordlist_steming)#剔除停用词
            file_wordlist.append(doc_wordlist)
            
        file_worddict.setdefault(key,file_wordlist) 
    #print('*********测试********')    
    #print(file_worddict[0][2])  
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    print("***************************预处理结束*****************************")
    return file_worddict

#####################################去掉高频和低频词#####################################

def wordfrequency(trainfile_worddict,low,high): 
    """
    计算文档词频并删除高频和低频词生成词典
    :param  :class_wordlist,low,high : 一类文章的单词列表，低频词下限和高频词上限
    :return :frequencydict : 生成词典
    """
    for key1 in trainfile_worddict.keys():
        wordlist1=list(_flatten(trainfile_worddict[key1]))#将二维列表降为一维列表 
        print('处理前第%d类所含的单词数为%d'%(key1,len(wordlist1)))
        frequencydict = dict(Counter(wordlist1))#统计所有文档中词的词频生成字典
        #返回的是一个是字典{'a': 2, 'c': 2, 'b': 2, 'd': 1} 
        record=[]
        for key in frequencydict.keys():
            if frequencydict[key]<low or frequencydict[key]>high: 
                record.append(key)
        for key in record:    
            frequencydict.pop(key)
        for docwordlist in trainfile_worddict[key1]:
            for word in docwordlist:
                if word not in list(frequencydict.values()) :
                    docwordlist.remove(word)
        wordlist2=list(_flatten(trainfile_worddict[key1]))
        print('处理后第%d类所含的单词数为%d'%(key1,len(wordlist2)))     
    return trainfile_worddict
                   
##################################NBC##############################

def NBC(trainfile_worddict,testfile_worddict,traindocnum): 
    """
    进行naive bayes 分类并计算正确率
    :param  :trainfile_worddict,testfile_worddict,traindocnum :训练集、测试集和训练集doc数量
    :return :acc : naive bayes 分类正确率
    """
    class_wordlist=[]
    frequencydict={}
    i=0
    j=0
    acc=0.0

    for key2 in testfile_worddict.keys():#取出测试集中每个类的嵌套列表
        testclass_plist=[]
        count=0
        for doc_wordlist in testfile_worddict[key2]:#取出测试集中每篇doc的词列表
            p_dict={}#储存训练集中一篇文章的每类对应的p值
            for key1 in trainfile_worddict.keys(): #取出训练集中一个类
                prior=numpy.log(len(trainfile_worddict[key1])/traindocnum)#计算当前类类的先验概率
                #print('第%d类的先验概率为%f'%(key1,len(trainfile_worddict[key1])/traindocnum))
                class_wordlist=list(_flatten(trainfile_worddict[key1]))#将二维列表降为一维列表 
                frequencydict = dict(Counter(class_wordlist))#统计所有文档中词的词频生成字典，返回的是一个是字典{'a': 2, 'c': 2, 'b': 2, 'd': 1} 
                total=len(class_wordlist)#计算训练集中当前类文档中所有词出现的次数（含重复）
                classwordnum=len(list(set(class_wordlist)))#计算训练集中当前类文档单词词典数量（不含重复）
                
                other_trainfile_worddict2 = trainfile_worddict.copy()#复制训练集
                other_trainfile_worddict2.pop(key1)#去掉训练集的当前类
                allotherclass_wordlist=[]
                for key3 in other_trainfile_worddict2.keys():#将去掉当前类的训练集每一类的中的二维列表降为一维列表 
                    eachclass_wordlist2=list(_flatten(other_trainfile_worddict2[key3]))
                    allotherclass_wordlist.append(eachclass_wordlist2)
                otherclass_wordlist=list(_flatten(allotherclass_wordlist))#将整个将去掉当前类的其他类训练集二维列表降为一维列表 
                other_total=len(otherclass_wordlist)#统计其他类训练集中单词的个数
                other_frequencydict = dict(Counter(otherclass_wordlist))#统计其他类训练集中词的词频生成字典
                other_classwordnum=len(list(set(otherclass_wordlist)))#计算训练集中每类文档单词词典数量（不含重复）
            
                #print('第%d类的N为%d,V为%d'%(key2,total,classwordnum1))
                p_eachdoc=0
                for each in doc_wordlist:#取出测试集中每篇doc的一个词
                    wordcount=frequencydict[each] if each in frequencydict.keys() else 0#当前词在当前类的词典里=frequencydict[each],当前词不在当前类的词典里=0
                    other_wordcount=other_frequencydict[each] if each in other_frequencydict.keys() else 0#当前词在其他类的词典里=frequencydict[each],当前词不在其他类的词典里=0
                  
                    #p_eachword=numpy.log((wordcount+1)/total)#不平滑计算
                    p_eachword_inclass=numpy.log((wordcount+1)/(total+classwordnum)) #平滑后的多项式模型计算每个单词在当前类的p值并取log  
                    p_eachword_notinclass=numpy.log((other_wordcount+1)/(other_total+other_classwordnum))#平滑后的多项式模型计算每个单词在其他类的p值并取log 
                    p_eachword=p_eachword_inclass-p_eachword_notinclass
                    p_eachdoc+=p_eachword#计算测试集中一篇文章的p值
                p_eachdoc+=prior
                p_dict.setdefault(key1,p_eachdoc)#将测试集中每个文档的P值放入字典
          
            p=max(p_dict,key=p_dict.get)#找出测试集中一篇文档最大的P值对应的类
            testclass_plist.append(p)#将测试集中一个类中每篇文章的bayes分类值放入列表
        print('*********测试开始********') 
        print('第%d类'%key2)
        print(testclass_plist)
        print(len(testclass_plist))#每一类的训练集doc数量
        #print(len(testfile_worddict[key2]))#每一类的训练集doc数量           
        for each in testclass_plist:
            if each==key2:
                count+=1
        n=count/len(testclass_plist)
           
        print('第%d类的正确率为%f'%(key2,n))
        print('*********测试结束********')  
        i+=count
        j+=len(testclass_plist)
    acc=i/j
    #print(j)
    print('The NBC result is %f'%acc ) 
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    print("*****************************NBC完毕*******************************")
    return acc

#####################################main函数####################################  
def main():
    """
    main函数调用上述函数完成整个程序
    :param  :
    :return :
    """  
    mainpath="C:\\Users\\311\\Desktop\\data mining\\20news-18828" 
    rate=0.2
    path_dict=getpath(mainpath,rate)
    trainpath_dict=path_dict[0]
    testpath_dict=path_dict[1]
    traindocnum=path_dict[2]
    trainfile_dict=readfile(trainpath_dict)
    testfile_dict=readfile(testpath_dict)
    trainfile_worddict0=preprocess(trainfile_dict)
    testfile_worddict=preprocess(testfile_dict)
    trainfile_worddict=wordfrequency(trainfile_worddict0,2,5000)
    acc=NBC(trainfile_worddict,testfile_worddict,traindocnum)
    return acc

#####################################end#################################### 
time=1
l=0.0
for t in range(time):
    acc=main()
    print('第%d次的The NBC result is %f'%(t+1,acc)) 
    l+=acc
ave=l/time
print('The NBC average result is %f'%ave)
  





   