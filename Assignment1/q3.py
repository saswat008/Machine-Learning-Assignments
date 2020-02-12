#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.patches as patches

#loading x(i) values from file into numpy array
def load_input(ipPath):
    ipData = np.genfromtxt(ipPath,delimiter=',')
    return ipData

#loading y(i) values from file into numpy array
def load_output(opPath):
    opData = np.genfromtxt(opPath,delimiter=',')
    return opData

#Normalizing X values
def normalizeX(x):
    x_mean = np.zeros(x.shape[1])
    x_std_dev = np.zeros(x.shape[1])
    
    for i in range(x.shape[1]):
            x_mean[i] = np.mean(x[:,i])
            x_std_dev[i] = np.std(x[:,i])
    
    for i in range(x.shape[1]):
        for j in range(x.shape[0]):
            x[j,i]= (x[j,i]-x_mean[i])/x_std_dev[i]
    return x

#Sigmoid Function
def sigmoidValue(x_theta):
    X_theta = -(x_theta)
    hox = 1/(1+np.exp(X_theta))
    return hox

#Calculating Hessian Matrix of J(theta)
def hessianMatrix(hox,X):
    D = np.diagflat(hox * (1 - hox))
    hessian = -(X.T @ D @ X)

    return hessian

#Calculating error in Newton's Method
def errorValue(X,Y,theta):
    x_Theta = X @ theta
    hox = sigmoidValue(x_Theta)
    diff = Y - hox
    
    grad_theta = X.T @ diff
    
    hessian = hessianMatrix(hox,X)
    h_Inverse = np.linalg.inv(hessian)
    
    error = h_Inverse @ grad_theta
    return error

#Predicting theta by implementing Newton's Method
def NewtonsMethod(X,Y,theta):
    converged = False
    theta_old = 0
    e = 1e-15
    steps = 0
    
    while (not converged):
        theta_old = theta
        theta-= errorValue(X,Y,theta)
        if (theta_old - theta).all() < e:
            converged = True
        steps+=1
    return theta


if __name__ == '__main__':
    ipPath = 'logisticX.csv'#input("Enter the input Path:")
    opPath = 'logisticY.csv'#input("Enter the output Path:")
    x = load_input(ipPath)
    y = load_output(opPath)

    m = x.shape[0]
    
    X = normalizeX(x)
    
    X = np.append(np.ones((m, 1)), X, axis = 1)
    Y = y.reshape(m,1)
    
    #Code for Part(a)
    theta = np.zeros((X.shape[1],1))
    theta = NewtonsMethod(X,Y,theta)
    
    X_Theta = np.dot(X,theta)
    hox = sigmoidValue(X_Theta)
    
    print('Theta = \n',theta)    


    #Code for Part(b)
    plt.figure(figsize=(6,6))
    ax=plt.subplot()
    plt.xlabel('X1')
    plt.ylabel('X2')

    l1 = 'Label0'
    l2 = 'Label1'

    for i in range(X.shape[0]):
        if Y[i] == 0:
            plt.scatter(X[i,1], X[i,2], color = 'blue',label = l1, marker='*')
            l1 = '_nolegend_'
        else:
            plt.scatter(X[i,1], X[i,2], color = 'orange',label = l2, marker='o')
            l2 = '_nolegend_'

    x_line = np.linspace(-2, 2, 4)
    y_line = (-1) * (x_line * theta[1] + theta[0])/(theta[2])

    # Plotting data points and Decision Boundary
    plt.plot(x_line, y_line, '-r', label='Decision Boundary')
    ax.legend(loc='upper right')
    plt.show()
