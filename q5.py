# EECS 545 HW3 Q5
# Your name: Han Yilun(yilunh)

# Install scikit-learn package if necessary:
# pip install -U scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import *


def readMatrix(filename: str):
    # Use the code below to read files
    with open(filename, 'r') as fd:
        hdr = fd.readline()
        rows, cols = [int(s) for s in fd.readline().strip().split()]
        tokens = fd.readline().strip().split()
        matrix = np.zeros((rows, cols))
        Y = []
        for i, line in enumerate(fd):
            nums = [int(x) for x in line.strip().split()]
            Y.append(nums[0])
            kv = np.array(nums[1:])
            k = np.cumsum(kv[:-1:2])
            v = kv[1::2]
            matrix[i, k] = v
        return matrix, tokens, np.array(Y)


def evaluate(output, label) -> float:
    # Use the code below to obtain the accuracy of your algorithm
    error = float((output != label).sum()) * 1. / len(output)
    return error

def main():
    # Load files
    # Note 1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note 2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    dataMatrix_train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q5_data/MATRIX.TEST')

    svc = LinearSVC(max_iter=100000)
    svc.fit(dataMatrix_train, category_train)
    #print('w：%s,b:%s'%(svc.coef_,svc.intercept_))
    print('trained parameter w = \n',svc.coef_)
    print('trained parameter b =',svc.intercept_)
    

    # Test and evluate
    prediction=svc.predict(dataMatrix_test)
    error = evaluate(prediction, category_test)
    print('Error: {:2.4f}%'.format(100 * error))

    error = np.zeros(6)
    datalist = [50,100,200,400,800,1400]
    for i in range(6):
        train, tokenlist, category_train = readMatrix('q5_data/MATRIX.TRAIN.'+ str(datalist[i]))
        svc = LinearSVC(max_iter=100000)
        svc.fit(train, category_train)
        #print('datasize %d: w = %s'%(datalist[i],w))
        #print('datasize %d: b = %s'%(datalist[i],b))

        # Test and evluate        
        prediction=svc.predict(dataMatrix_test)
        cnt = sum(abs((train@(svc.coef_.T)+svc.intercept_))<=1)[0]
        error[i]=evaluate(prediction, category_test)
        print('Error in dateset of size {}: {:2.4f}% with # SVs = {}'.format(datalist[i],100*error[i],cnt))
        
    
    plt.xlabel("size of datasets")
    plt.ylabel("error")
    plt.plot(datalist,error, color = 'r')
    plt.show()

if __name__ == '__main__':
    main()
