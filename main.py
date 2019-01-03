from mlearn import *
import numpy as np
if __name__ == '__main__':

    testMat = np.mat(np.eye(4))

    regAlg = regTree()
    mat0, mat1 = regAlg.spliteData(testMat, 1, 0.5)

    tool = tools()
    dataSet = tool.loadData('ex00.txt')

    print('stop mlearn...')

