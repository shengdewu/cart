import numpy as np

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
        regTree['spInd'] = bestIndex
        regTree['spVal'] = bestVal
        left, right = self.spliteData(dataSet, bestIndex, bestVal)
        regTree['left'] = self.createTree(left, errMeth, leafMeth, ops)
        regTree['right'] = self.createTree(right, errMeth, leafMeth, ops)
        return regTree
