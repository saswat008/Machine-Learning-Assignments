#!/usr/bin/env python
# coding: utf-8

import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Function for shuffling data
def shuffleData(x,y):
    data = np.append(x,y,axis=1)
    np.random.shuffle(data)
    X = data[:,:3]
    Y = data[:,3:]
    return X,Y

#Function for calculating cost of a batch
def calculateCost(X,Y,theta,r):
    return (1 / (2*r)) * np.sum((Y - np.dot(X, theta)) ** 2)

#Calculating the gradient of J(theta)
def grad_descent(X,Y,theta):
    deltaJ = np.dot(X,theta) - Y
    tmp = X.T @ deltaJ
    return tmp


if __name__ == '__main__':
    #Code for Part(a)
    x1_mu = 3
    x1_sigma = math.sqrt(4)
    x1 = np.random.normal(x1_mu, x1_sigma, 1000000)
    X1 = x1.reshape(x1.shape[0],1)
    
    x2_mu = -1
    x2_sigma = math.sqrt(4)
    x2 = np.random.normal(x2_mu, x2_sigma, 1000000)
    X2 = x2.reshape(x2.shape[0],1)
    
    epsilon_mu = 0
    epsilon_sigma = math.sqrt(2)
    epsilon = np.random.normal(epsilon_mu, epsilon_sigma, 1000000)
    Epsilon = epsilon.reshape(epsilon.shape[0],1)
    
    theta = np.array([3,1,2])
    theta = theta.reshape(3,1)
    
    x = np.append(np.append(np.ones((1000000,1)), X1, axis = 1), X2, axis = 1)
    y0 = (x @ theta)
    y = y0 + Epsilon
    #sns.distplot(Y)


    #Code for Part(b)
    m = x.shape[0]
    eta = 1e-3
    thetaL = np.zeros((theta.shape[0],1))
    thetaList = thetaL
    X,Y = shuffleData(x,y)
    batchSize = 1
    numOfBatches = m//batchSize
    Jtheta = 0
    JthetaSum = 0
    avgCost = 0
    avgCost_old = 0
    it = 0

    while True:
        for batch in range(numOfBatches):
            batchStart = batch*batchSize
            batchEnd = (batch+1)*batchSize
            batchX = X[batchStart:batchEnd,:]
            batchY = Y[batchStart:batchEnd,:]
    
            Jtheta = calculateCost(batchX,batchY,thetaL,batchSize)
            JthetaSum+=Jtheta
            thetaL-= (eta * grad_descent(batchX,batchY,thetaL))/batchSize
            thetaList = np.append(thetaList,thetaL,axis = 1)
            it+=1
            if batch == numOfBatches-1 or it%1000==0:
                avgCost_old = avgCost
                avgCost = JthetaSum/1000
                JthetaSum = 0
                if abs(avgCost_old - avgCost) < 1e-3:
                    break
        if abs(avgCost_old - avgCost) < 1e-3:
            break
        
    print('Theta=\n',thetaL)
    #print(calculateCost(X,Y,thetaL,m))

    #Code for Part(c)
    filePath = 'q2test.csv'
    data = np.genfromtxt(filePath,delimiter=',')
    data = data[1:,:]
    testX = np.append(np.ones((data.shape[0],1)),data[:,:2],axis=1)
    testY = data[:,2:]

    testError = calculateCost(testX,testY,thetaL,data.shape[0])
    testErrorOriginal = calculateCost(testX,testY,theta,data.shape[0])
    print('Test Error with Learnt parameters: ',testError,'\nTest Error with Original Parameters: ', testErrorOriginal)


    #Code for Part (d)
    thetaList = thetaList.T
    t0 = thetaList[:,0]
    t1 = thetaList[:,1]
    t2 = thetaList[:,2]

    from mpl_toolkits.mplot3d.axes3d import Axes3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('Theta_0')
    ax.set_ylabel('Theta_1')
    ax.set_zlabel('Theta_2')
    ax.plot(t0,t1,t2,'r.')
    plt.show()
