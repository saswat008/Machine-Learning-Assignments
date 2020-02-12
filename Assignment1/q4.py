#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.patches as patches

#Normalize the x(i) values
def normalizeX(x):
    x_mean = np.zeros((x.shape[1],1))
    x_std_dev = np.zeros((x.shape[1],1))
    for i in range(x_mean.shape[0]):
        x_mean[i] = np.mean(x[:,i])
        x_std_dev[i] = np.std(x[:,i])
    
    for i in range(x.shape[0]):
        x[i,:]= (x[i,:]-x_mean.T)/x_std_dev.T
        
    return x

if __name__ == '__main__':
    x = np.loadtxt('q4x.dat')
    
    y = np.loadtxt('q4y.dat',dtype=np.str)
    y = y.reshape(y.shape[0],1)
    
    data = np.append(x, y, axis=1)
    m = data.shape[0]
    
    count0 = 0
    count1 = 0
    
    x = normalizeX(x)
    
    #Code for Part(a)
    x0Sum = np.array(np.zeros((1,x.shape[1]),dtype='float'))
    x1Sum = np.array(np.zeros((1,x.shape[1]),dtype='float'))
    
    #0 for Alaska; 1 for Canada
    for i in range(data.shape[0]):
        if y[i] == 'Alaska':
            x0Sum+= (x[i])
            count0+=1
        else:
            x1Sum+= (x[i])
            count1+=1
    
    mu0 = x0Sum/count0
    
    mu1 = x1Sum/count1
    
    phi = count1/m
    
    xDiff = [x[i]-mu0 if y[i]=='Alaska' else x[i]-mu1 for i in range(x.shape[0])]
    xDiff = (np.asarray(xDiff)).reshape(m,2)
    
    sigma  = (xDiff.T @ xDiff)/m
    print('mu0 = ',mu0,'\nmu1 = ',mu1,'\nsigma = \n',sigma)        


    #Code for Part(b)
    plt.figure(figsize=(8,8))
    l1 = 'Alaska'
    l2 = 'Canada'

    for i in range(data.shape[0]):
        if y[i] == 'Alaska':
            plt.scatter(x[i,0], x[i,1], color = 'green',label = l1, marker='*')
            l1 = '_nolegend_'
        else:
            plt.scatter(x[i,0], x[i,1], color = 'yellow',label = l2,marker='o')
            l2 = '_nolegend_'
    plt.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')

    #Code for Part(c)
    theta0L = np.log(phi/(1-phi)) - 0.5*((mu1 @ np.linalg.inv(sigma) @ mu1.T) - (mu0 @ np.linalg.inv(sigma) @ mu0.T))
    theta1L = np.linalg.inv(sigma) @ (mu1 - mu0).T
    thetaL = np.append(theta0L,theta1L,axis=0)

    lx_line = np.linspace(-2, 2, 15)
    ly_line = (-1) * (thetaL[1]*lx_line + thetaL[0])/thetaL[2]

    plt.plot(lx_line,ly_line,'-r',label='Decision Boundary')
    plt.legend()
    plt.show()

    #Code for Part(d)
    x0Diff = np.zeros((1,x.shape[1]))
    x1Diff = np.zeros((1,x.shape[1]))

    for i in range(x.shape[0]):
        if y[i]=='Alaska':
            x0Diff = np.append(x0Diff,x[i]-mu0,axis=0)
        else:
            x1Diff = np.append(x1Diff,x[i]-mu1,axis=0)

    x0Diff = x0Diff[1:,:]
    x1Diff = x1Diff[1:,:]

    sigma0 = (x0Diff.T @ x0Diff)/count0
    sigma1 = (x1Diff.T @ x1Diff)/count1

    print('Sigma0 =\n',sigma0,'\nSigma1 =\n',sigma1)

    #Code for Part(e)
    theta0 = np.log(phi/(1-phi)) + np.log((math.sqrt(np.linalg.det(sigma0)))/(math.sqrt(np.linalg.det(sigma1)))) - 0.5*((mu1 @ np.linalg.inv(sigma1) @ mu1.T) - (mu0 @ np.linalg.inv(sigma0) @ mu0.T))
    theta1 = (np.linalg.inv(sigma1) @ mu1.T) - (np.linalg.inv(sigma0) @ mu0.T)
    theta2 = 0.5 * (np.linalg.inv(sigma0) - np.linalg.inv(sigma1))
    thetafr = np.append(theta0, theta1.T, axis = 1)
    theta = np.append(thetafr, np.append(theta1, theta2, axis =1), axis = 0)

    l1 = 'Alaska'
    l2 = 'Canada'

    qx_line = np.linspace(-2, 2, 15)

    a = theta[2][2]
    b = (theta[1][2] + theta[2][1])*qx_line + theta[0][2]
    c = (theta[1][1] * qx_line**2) + theta[0][1]*qx_line + theta[0][0]

    qy1line = np.array(np.zeros(qx_line.shape[0]))
    #qy2line = np.array(np.zeros(qx_line.shape[0]))

    for i in range(b.shape[0]):
        qy1line[i] = ((-1)*b[i] - math.sqrt(b[i]**2 - (4*a)*c[i]))/(2*a)
        #qy2line[i] = ((-1)*b[i] + math.sqrt(b[i]**2 - (4*a)*c[i]))/(2*a)

    plt.figure(figsize=(10,10))

    for i in range(data.shape[0]):
        if y[i] == 'Alaska':
            plt.scatter(x[i,0], x[i,1], color = 'green', label = l1, marker='*')
            l1 = '_nolegend_'
        else:
            plt.scatter(x[i,0], x[i,1], color = 'yellow', label = l2, marker='o')
            l2 = '_nolegend_'

    plt.legend()
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.plot(lx_line,ly_line,'-b',label='Linear Decision Boundary')
    plt.plot(qx_line,qy1line,'-r',label='Quadratic Decision Boundary')
    plt.legend()
    #plt.plot(qx_line,qy2line,'-r')
    plt.show()

