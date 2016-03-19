# -*- coding:utf8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
'''
kNN分类算法
'''

def createDataSet():
    '''
    创造数据集
    :return: 数据集及标签
    '''
    group = np.array([1.0, 1.1, 1.0, 1.0, 0, 0, 0, 0.1])
    group.shape = 4, 2
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    '''

    :param inX:未知点
    :param dataSet:数据集
    :param labels:标签，类别
    :param k: 近邻数
    :return: 标签，类别
    '''
    #数据集大小
    dataSetSize = dataSet.shape[0]
    inX = np.array(inX)
    diff = inX - dataSet
    distances = np.sqrt(np.sum(diff**2, axis=1))
    #返回排序后索引
    sortedDistIndexs = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndexs[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    #对字典排序
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1], reverse=True)
    return sortedClassCount[0][0]

def file2dataSet(filename):
    '''
    从特定格式的文件中读取数据集
    :param filename: 文件名
    :return: 数据集 标签
    '''
    f = open(filename, 'r')
    lines = [lines.strip().split('\t') for lines in f.readlines()]
    f.close()
    #数据集大小
    dataSetSize = len(lines)
    #特征量维度
    dim = len(lines[0]) - 1
    dataSet = np.zeros((dataSetSize, dim))
    labels = []
    for line, i in zip(lines, range(dataSetSize)):
        dataSet[i,:] = line[0:dim]
        labels.append(int(line[-1]))
    return dataSet, labels

def autoNorm(dataSet):
    '''
    归一化
    '''
    minVals = np.min(dataSet, axis=0)
    maxVals = np.max(dataSet, axis=0)
    ranges = maxVals - minVals
    normDataSet = (dataSet - minVals)/ranges
    return normDataSet, ranges, minVals
def datingClassTest():
    '''
    用于测定算法的精确度
    '''
    hoRatio = 0.1
    datingDataMat, datingLables = file2dataSet('datingTestSet2.txt')
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(norMat[i,:], norMat[numTestVecs:m,:], datingLables[numTestVecs:m],3)
        print("分类结果：{0}，而实际的类别是：{1}".format(classifierResult, datingLables[i]))
        if classifierResult != datingLables[i]:
            errorCount += 1
    return errorCount/numTestVecs
def image2vector(filename):
    '''
    将图像转化成向量32*32  - 1*1024
    '''
    f = open(filename, 'r')
    returnVec = np.zeros(1024)
    lines = (x.strip() for x in f.readlines())
    for i, line in zip(range(32), lines):
        returnVec[32*i:32*(i+1)] = list(map(int,line))
    f.close()
    return returnVec

def handwritingClassTest():
    '''
    用于测试文字识别
    '''
    hwLabels = []
    #获取训练集文件列表
    trainingFileList = os.listdir('trainingDigits')
    size = len(trainingFileList)
    trainingMat = np.zeros((size, 1024))
    for i in range(size):
        fileName = trainingFileList[i]
        fileStr = fileName.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        hwLabels.append(classNum)
        trainingMat[i,:] = image2vector('trainingDigits/{0}'.format(fileName))
    #获取测试集文件列表
    testFileList = os.listdir('testDigits')
    testSize = len(testFileList)
    errorCount = 0
    for i in range(testSize):
        filename = testFileList[i]
        fileStr = filename.split('.')[0]
        classNum = int(fileStr.split('_')[0])
        testVec = image2vector('testDigits/{0}'.format(filename))
        classifierResult = classify0(testVec, trainingMat, hwLabels, 3)
        print("分类结果：{0}, 实际类别：{1}".format(classifierResult, classNum))
        if classifierResult != classNum:
            errorCount += 1
    print("分类错误数：{0}".format(errorCount))
    print('分类错误率:{0:.4f}'.format(errorCount/testSize))


if __name__ == '__main__':
    '''
    x = [0, 0]
    dataSet, labels = createDataSet()
    category = classify0(x, dataSet, labels, 3)
    print("类别", category)
    dataSet, labels = file2dataSet('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = 'rgb'
    markers = 'o*s'
    legends = ['Did Not like', 'Liked in Small Doses', 'Likess in Large Doses']
    for i in range(3):
        index = np.array(labels) == i+1
        ax.scatter(dataSet[index,0], dataSet[index,1],marker=markers[i], s=30,c=colors[i],label=legends[i])
    plt.legend(loc=2)
    plt.show()
    '''

    '''
    dataSet, labels = file2dataSet('datingTestSet2.txt')
    print(dataSet)
    normDataSet, ranges, minVals = autoNorm(dataSet)
    print(normDataSet.shape)
    print(ranges)
    print(minVals)
    '''
    handwritingClassTest()

