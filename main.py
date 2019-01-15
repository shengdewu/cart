from mlearn import *
import numpy as np

if __name__ == '__main__':

    testMat = np.mat(np.eye(4))

    regAlg = regTree()
    mat0, mat1 = regAlg.spliteData(testMat, 1, 0.5)

    #均值误差法
    # tool = tools()
    # dataSet = tool.loadData('./testData/ex2.txt')
    # tree = regAlg.createTree(dataSet, tool.errCalc, tool.leafCreate, (0, 1))
    #
    # print(str('regtree {}').format(tree))
    # dataTest = tool.loadData('./testData/ex2test.txt')
    # ptest = regAlg.prun(tree, dataTest)
    # print(str('purn {}').format(ptest))

    #线性方程法
    # tool = tools()
    # dataSet = tool.loadData('./testData/exp2.txt')
    # tree = regAlg.createTree(dataSet, tool.errMode, tool.leafMode, (1, 10))
    #
    # print(str('regtree {}').format(tree))

    tool = tools()
    dataTest = tool.loadData('./testData/bikeSpeedVsIq_test.txt')
    dataTrain = tool.loadData('./testData/bikeSpeedVsIq_train.txt')
    regTree = regAlg.createTree(dataTrain, tool.errReg, tool.leafReg, (1, 20))
    modeTree = regAlg.createTree(dataTrain, tool.errMode, tool.leafMode, (1, 20))

    print(str('regtree {} \n').format(regTree))
    RY = regAlg.predict(regTree, dataTest[:,0], tool.preReg)
    print(str('regpre {} \n').format(np.corrcoef(RY, dataTest[:,1], rowvar=0)[0,1]))

    print(str('modetree {} \n').format(modeTree))
    MY = regAlg.predict(modeTree, dataTest[:,0], tool.preMode)
    print(str('modepre {} \n').format(np.corrcoef(MY, dataTest[:, 1], rowvar=0)[0, 1]))
    print('stop mlearn...')

