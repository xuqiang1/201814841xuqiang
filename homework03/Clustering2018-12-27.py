# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:13:55 2018

@author: 311
"""

import json
import random
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.cluster import normalized_mutual_info_score

from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
#from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


##################################读入文本划分训练和测试数据集#####################

def readfile(mainpath):
    """
    按行读出文档并储存在列表的结构中
    :param  :mainpath : 
    :return :text_list, text_label : 将数据和标签分为两个列表储存
    """
    line_dict={}
    text_list=[]
    reallabel_list=[]
    
    f = open(mainpath,"r")    
    for line in f.readlines():
        line_dict = json.loads(line)
        text_list.append(line_dict["text"])
        reallabel_list.append(line_dict["cluster"])          
    #print(len(text_list),len(reallabel_list)) 
    print("***************************文件读取完成*****************************")
    return text_list, reallabel_list 

 ######################################计算TF-IDF矩阵###############################   

def tfidf_matrix(text_list):
    """
    计算所有文档的TF-IDF生成矩阵
    :param  :text_list: filelist文档列表
    :return :tfidf_weight：一个由TF-IDF组成的矩阵，元素a[i][j]表示j词在i类文本中的TF-IDF权重
    """
    vectorizer = CountVectorizer(stop_words="english")#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(vectorizer.fit_transform(text_list))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    #print(tfidf)
    tfidf_weight = tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    #print(tfidf_weight)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    return tfidf_weight

 ######################################K-means#####################################  

def kmeans(tfidf_weight, reallabel_list):
    """
    用K-means进行聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """
    r=random.randint(0,99)
    clustering = KMeans(n_clusters=110, random_state=r).fit(tfidf_weight)#设置K值为110，随机数为0-99
    predictlabel_list = clustering.labels_
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    return score

 ######################################Affinity Propogation#########################
 
def affinity_propagation(tfidf_weight, reallabel_list):
    """
    用Affinity Propogation进行聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """
    clustering = AffinityPropagation(damping=0.95).fit(tfidf_weight)
    predictlabel_list = clustering.labels_ 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')    
    return score

 ########################################Mean-Shift#################################
 
def mean_shift(tfidf_weight, reallabel_list):
    """
    均值迁移聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """
    #bandwidth_c = estimate_bandwidth(tfidf_weight, quantile=0.2, n_samples=200)#bandwidth ：半径(或带宽)，float型，使用sklearn.cluster.estimate_bandwidth计算出半径(带宽).
    #print(bandwidth_c)
    clustering = MeanShift(bandwidth=0.6, bin_seeding=True).fit(tfidf_weight)#如果为真，初始内核位置不是所有点的位置，而是点的离散版本的位置，其中点被分类到其粗糙度对应于带宽的网格上。将此选项设置为True将加速算法，因为较少的种子将被初始化。
    predictlabel_list = clustering.labels_ 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')    
    return score

 ########################################Spectral Clustering########################
 
def spectral_clustering(tfidf_weight, reallabel_list):
    """
    谱聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """    
    clustering = SpectralClustering(n_clusters=110, assign_labels='discretize', random_state=0).fit(tfidf_weight)
    predictlabel_list = clustering.labels_ 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')    
    return score

 ########################################Agglomerative Clustering########################
 
def agglomerative_clustering(tfidf_weight, reallabel_list,linkage):
    """
    层次聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list,linkage：一个由TF-IDF组成的矩阵，真实标签列表，计算组间距离的方法
    :return :score：用NMI来评估预测结果
    """     
    clustering = AgglomerativeClustering(n_clusters=110, linkage=linkage).fit(tfidf_weight)#linkage：指定层次聚类判断相似度的方法
    predictlabel_list = clustering.labels_ 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')    
    return score

 #############################################DBSCAN#####################################
 
def dbscan(tfidf_weight, reallabel_list):
    """
   密度聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list,linkage：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """     
    clustering = DBSCAN(eps=1.09, min_samples=2).fit(tfidf_weight)#邻居最大距离为1.09，形成簇的最小样本数
    predictlabel_list = clustering.labels_ 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')    
    return score

 #############################################Gaussian Mixtures##########################
 
def gaussian_mixture(tfidf_weight, reallabel_list):
    """
    GMM聚类，并将预测结果和真实值作对比
    :param  :tfidf_weight, reallabel_list,linkage：一个由TF-IDF组成的矩阵，真实标签列表
    :return :score：用NMI来评估预测结果
    """     
    clustering = GaussianMixture(n_components=110, covariance_type='tied', random_state=0).fit(tfidf_weight)# covariance_type协方差类型，tied：所用模型共享一个一般协方差矩阵
    predictlabel_list = clustering.predict(tfidf_weight) 
    score = normalized_mutual_info_score(predictlabel_list, reallabel_list)
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')#现在
    print(nowTime,'\n')
    return score

 ######################################main函数###########################################
   
def main():
    """
    main函数调用上述函数完成整个程序
    :param  :
    :return :
    """  
    mainpath="C:\\Users\\311\\Desktop\\data mining\\201814841xuqiang\\homework03\\Tweets.txt" 
    text_list,reallable_list=readfile(mainpath)
    tfidf_weight=tfidf_matrix(text_list)
    
    print("The K-Means score is:%f"%(kmeans(tfidf_weight,reallable_list)))
    print("The Affinity Propogation score is:%f"%(affinity_propagation(tfidf_weight,reallable_list)))
    print("The Mean-Shift score is:%f"%(mean_shift(tfidf_weight,reallable_list)))
    print("The Spectral Clustering score is:%f"%(spectral_clustering(tfidf_weight,reallable_list)))
    print("The Agglomerative Clustering-average score is:%f"%(agglomerative_clustering(tfidf_weight,reallable_list, 'average')))#average：组间距离等于两组对象之间的平均距离（average-linkage聚类）
    print("The DBSCAN score is:%f"%(dbscan(tfidf_weight,reallable_list)))
    print("The Gaussian Mixtures score is:%f"%(gaussian_mixture(tfidf_weight,reallable_list)))

########################################end#############################################
main()


   