from mlearn import *
import numpy as np
if __name__ == '__main__':

    testMat = np.mat(np.eye(4))

    regAlg = regTree()
    mat0, mat1 = regAlg.spliteData(testMat, 1, 0.5)

    tool = tools()
    dataSet = tool.loadData('./testData/ex2.txt')
    tree = regAlg.createTree(dataSet, tool.errCalc, tool.leafCreate, (0, 1))

    print(str('regtree {}').format(tree))
    dataTest = tool.loadData('./testData/ex2test.txt')
    ptest = regAlg.prun(tree, dataTest)
    print(str('purn {}').format(ptest))
    print('stop mlearn...')

