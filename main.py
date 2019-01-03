from mlearn import *
import numpy as np
if __name__ == '__main__':

    testMat = np.mat(np.eye(4))

    regAlg = regTree()
    mat0, mat1 = regAlg.spliteData(testMat, 1, 0.5)

    tool = tools()
    dataSet = tool.loadData('./testData/ex00.txt')
    tree = regAlg.createTree(dataSet, tool.errCalc, tool.leafCreate, (1, 4))

    print(str('regtree {}').format(tree))

    print('stop mlearn...')

