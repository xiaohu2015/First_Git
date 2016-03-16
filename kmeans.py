# -*- coding:utf-8 -*-
'''
K-means算法
'''
import numpy as np
import matplotlib.pyplot as plt
#获取数据集信息
def loadDataSet(filename):
    dataSet = []
    f = open(filename, 'r')
    lines = (x.strip() for x in f.readlines())
    for line in lines:
        curLine = line.split('\t')
        fltLine = map(float, curLine)
        dataSet.append(list(fltLine))
    return np.array(dataSet)

def normalization(dataset):
    '''
    归一化处理
    :param dataset: 数据集
    :return:
    '''
    dMax = np.max(dataset, axis=0)
    dMin = np.min(dataset, axis=0)
    newDataSet = (dataset - dMin)/(dMax - dMin)
    return newDataSet

def distEclud(vecA, vecB):
    '''
    计算距离，采用欧几里得距离
    :param vecA: 向量A
    :param vecB: 向量B
    :return:
    '''
    return np.sqrt(np.sum((vecA - vecB)**2))

def randCent(dataSet, k):
    '''
    用于产生一个随机质心
    :param dataSet:数据集
    :param k:分类数目
    :return:
    '''
    n = dataSet.shape[1]  #获取数据集特征参数总数
    centroids = np.zeros((k, n))   #初始化质心
    for i in range(n):
        minJ = np.min(dataSet[:,i])
        maxJ = np.max(dataSet[:,i])
        centroids[:,i] = minJ + (maxJ - minJ)*np.random.rand(k)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    '''
    主函数
    :param dataSet: 数据集
    :param k: 分类数
    :param distMeas: 计算距离方法，默认为distEclud
    :param createCent: 随机产生质心方法
    :return:
    '''
    m = dataSet.shape[0]   #数据集大小
    clusterAssment = np.zeros((m,2))  #存储每个数据点的聚类信息
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist, minIndex = distJI, j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
                clusterAssment[i,:] = minIndex, minDist**2
            print('质心', centroids)
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment[:,0]==cent, :]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

if __name__ == '__main__':
    dataSet = loadDataSet('data.txt')
    #dataSet = normalization(dataSet)
    centroids, clusterAssment = kMeans(dataSet, 4)
    print(centroids)
    print(clusterAssment)
    plt.figure()
    colors = 'rgbc'
    markers = 'oD*s'
    for i in range(4):
        data = dataSet[clusterAssment[:,0]==i,:]
        plt.scatter(data[:,0], data[:,1],c=colors[i],marker=markers[i],s=30)
    plt.show()

