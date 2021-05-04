import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    # Use the code below to read files
    fd = open(file, 'r')
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

def nb_train(matrix, category):
    # Implement your algorithm and return
     
    ############################
    # Implement your code here #
    nsp_email = matrix[category == 0]
    sp_email = matrix[category == 1]
    Nspam = sp_email.shape[0]
    Nnspam = nsp_email.shape[0]
    Njspam = np.sum(sp_email,axis=0)
    Njnspam = np.sum(nsp_email,axis=0)

    phi = Nspam/(Nspam+Nnspam)
    Miuj_s = (Njspam + 1)/(sum(Njspam)+len(Njspam))
    Miuj_ns = (Njnspam + 1)/(sum(Njnspam)+len(Njnspam))
    state = dict(phi = phi, Miu_s = Miuj_s, Miu_ns = Miuj_ns)

    return state

def nb_test(matrix, state):
    # Classify each email in the test set (each row of the document matrix) as 1 for SPAM and 0 for NON-SPAM
    output = np.zeros(matrix.shape[0])
    
    ############################
    # Implement your code here #
    phi = state['phi']
    Miu_s = state['Miu_s']
    Miu_ns = state['Miu_ns']
   
    for i in range(matrix.shape[0]):
        sp = sum(matrix[i] * np.log(Miu_s)) +np.log(phi)
        nsp = sum(matrix[i] * np.log(Miu_ns)) +np.log(1-phi)
        if sp>nsp:
            output[i] = 1
        else:
            output[i] = 0
    #print('predicted =') 
    #print(output)   

    return output

def evaluate(output, label):
    # Use the code below to obtain the accuracy of your algorithm
    error = (output != label).sum() * 1. / len(output)
    #print('Error: {:2.4f}%'.format(100*error))
    return error

def main():
    # Note1: tokenlists (list of all tokens) from MATRIX.TRAIN and MATRIX.TEST are identical
    # Note2: Spam emails are denoted as class 1, and non-spam ones as class 0.
    # Note3: The shape of the data matrix (document matrix): (number of emails) by (number of tokens)

    # Load files
    dataMatrix_train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN')
    dataMatrix_test, tokenlist, category_test = readMatrix('q4_data/MATRIX.TEST')

    # Train
    state = nb_train(dataMatrix_train, category_train)

    # Test and evluate
    prediction = nb_test(dataMatrix_test, state)
    error = evaluate(prediction, category_test)
    print('Error: {:2.4f}%'.format(100*error))

    # most indicative tokens
    indicator = np.log(state['Miu_s']/state['Miu_ns'])
    max5 = np.argsort(-indicator)[0:5]
    print('Most indicative 5 tokens: %s, %s, %s, %s, %s'%(tokenlist[max5[0]],tokenlist[max5[1]],tokenlist[max5[2]],tokenlist[max5[3]],tokenlist[max5[4]]))

    #other datasets
    datalist = [50,100,200,400,800,1400]
    error = np.zeros(6)
    for i in range(6):
        train, tokenlist, category_train = readMatrix('q4_data/MATRIX.TRAIN.'+ str(datalist[i]))
        # Train
        state = nb_train(train, category_train)

        # Test and evluate
        prediction = nb_test(dataMatrix_test, state)
        error[i] = evaluate(prediction, category_test)
        print('Error in dateset of size {}: {:2.4f}%'.format(datalist[i],100*error[i]))
    
    print('the best error is in the dataset of size %d'%datalist[np.argsort(error)[0]])
    
    plt.xlabel("size of datasets")
    plt.ylabel("error")
    plt.plot(datalist,error, color = 'r')
    plt.show()
        

if __name__ == "__main__":
    main()
        
