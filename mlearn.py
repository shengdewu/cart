import numpy as np
from enum import Enum

class treeInfo(Enum):
    LEFT='left'
    RIGHT='right'
    SPINDEX='spInd'
    SPVAL='spVal'

class treeNode(object):
    def __init__(self, feat, value, left, right):
        '''
        :param feat: 划分的特征
        :param value:划分的特征值
        :param left:左子树
        :param right:右子树
        '''
        self.featFeature = feat
        self.splitValue = value
        self.leftTree = left
        self.rightTree = right
        return

class tools(object):
    def loadData(self, path):
        dataMat = []
        with open(path, 'r') as df:
            for line in df.readlines():
                data = map(float, line.strip('\n').split('\t'))
                dataMat.append(list(data))

        return np.mat(dataMat)

    def errCalc(self, dataSet):
        return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

    def leafCreate(self, dataSet):
        return np.mean(dataSet[:,-1])

    def sloveLine(self, dataSet):
        #这种 y = x * w + w 考虑了正则，增加了误差项
        m, n = np.shape(dataSet)
        X = np.mat(np.ones((m, n)))
        Y = np.mat(np.ones((m, 1)))
        X[:, 1:n] = dataSet[:, 0:n - 1]
        Y = dataSet[:, -1]

        #这种 y = x * w
        # X = dataSet[:,0:-1]
        # Y = dataSet[:,-1]
        XTX = X.T * X
        if 0 == np.linalg.det(XTX):
            raise ValueError(str('the matrix is not Reversible'))
        w = np.linalg.inv(XTX) * (X.T * Y)
        return w, X, Y

    def errMode(self, dataSet):
        w, X, Y = self.sloveLine(dataSet)
        return np.sum(np.power(Y-X*w, 2))

    def leafMode(self, dataSet):
        w, X, Y = self.sloveLine(dataSet)
        return w

class regTree(object):

    def spliteData(self, dataSet, feature, value):
        mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
        mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
        return mat0, mat1

    def chooseBestFeet(self, dataSet, errMeth=None, leafMeth=None, ops=()):
        '''
        选择特征
        :param dataSet: (X,Y） dataSet[:,-1] 是值 等价于分类中的类别
        :param errMeth: 误差计算方法
        :param leafMeth: 叶子节点的方法
        :param ops: 可选参选
        :return: 特征索引，特征值
        '''

        #如果类别一样则不划分
        if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
            return None, leafMeth(dataSet)

        fdim, fnum = np.shape(dataSet)
        bestIndex = None
        initErr = errMeth(dataSet)
        bestError = np.inf
        bestVal = None
        for index in range(fnum-1): #跳过标签
            for splitVal in set(dataSet[:,index].T.tolist()[0]):
                mat0, mat1 = self.spliteData(dataSet, index, splitVal)
                if np.shape(mat0)[0] < ops[1] or np.shape(mat1)[0] < ops[1]:
                    continue
                sumErr = errMeth(mat0) + errMeth(mat1)
                if bestError > sumErr:
                    bestError = sumErr
                    bestIndex = index
                    bestVal = splitVal

        #误差变化太小则忽略
        if initErr - bestError < ops[0]:
            return None, leafMeth(dataSet)
        mat0, mat1 = self.spliteData(dataSet, bestIndex, bestVal)
        if np.shape(mat0)[0] < ops[1] or np.shape(mat1)[0] < ops[1]:
            return None, leafMeth(dataSet)
        return bestIndex, bestVal

    def createTree(self, dataSet, errMeth=None, leafMeth=None, ops=()):
        bestIndex, bestVal = self.chooseBestFeet(dataSet, errMeth, leafMeth, ops)
        if bestIndex == None:
            return bestVal
        regTree = {}
        regTree[treeInfo.SPINDEX.value] = bestIndex
        regTree[treeInfo.SPVAL.value] = bestVal
        left, right = self.spliteData(dataSet, bestIndex, bestVal)
        regTree[treeInfo.LEFT.value] = self.createTree(left, errMeth, leafMeth, ops)
        regTree[treeInfo.RIGHT.value] = self.createTree(right, errMeth, leafMeth, ops)
        return regTree

    def isTree(self, tree):
        return isinstance(tree, dict)

    def getMean(self, tree):
        if not self.isTree(tree):
            return tree

        if self.isTree(tree[treeInfo.LEFT.value]):
            tree[treeInfo.LEFT.value] = self.getMean(tree[treeInfo.LEFT.value])

        if self.isTree(tree[treeInfo.RIGHT.value]):
            tree[treeInfo.RIGHT.value] = self.getMean(tree[treeInfo.RIGHT.value])

        return (tree[treeInfo.LEFT.value] + tree[treeInfo.RIGHT.value])/2

    def prun(self, tree, testData):
        if np.shape(testData)[0] == 0:
            return self.getMean(tree)

        if not self.isTree(tree):
            return tree

        if self.isTree(tree[treeInfo.LEFT.value]) or self.isTree(tree[treeInfo.RIGHT.value]):
            left, right = self.spliteData(testData, tree[treeInfo.SPINDEX.value], tree[treeInfo.SPVAL.value])
            tree[treeInfo.LEFT.value] = self.prun(tree[treeInfo.LEFT.value], left)
            tree[treeInfo.RIGHT.value] = self.prun(tree[treeInfo.RIGHT.value], right)

        if (not self.isTree(tree[treeInfo.LEFT.value])) and (not self.isTree(tree[treeInfo.RIGHT.value])):
            left, right = self.spliteData(testData, tree[treeInfo.SPINDEX.value], tree[treeInfo.SPVAL.value])
            lvar = 0; rvar=0
            if len(left):
                lvar = np.var(left[:,-1]) * np.shape(left)[0]
            if len(right):
                rvar = np.var(right[:,-1]) * np.shape(right)[0]
            noMergeErr = np.power(lvar - tree[treeInfo.LEFT.value], 2) + np.power(rvar - tree[treeInfo.RIGHT.value],2)
            treeMean = (tree[treeInfo.LEFT.value] + tree[treeInfo.RIGHT.value])/2
            mergeErr = np.power(np.var(testData[:,-1]) * np.shape(testData)[0] - treeMean,2)

            if noMergeErr > mergeErr:
                print('merge...')
                return treeMean

        return tree
