#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.axes3d import Axes3D

#loading x(i) values from file into numpy array
def load_input(ipPath):
    ipData = np.genfromtxt(ipPath,delimiter=',')
    return ipData

#loading y(i) values from file into numpy array
def load_output(opPath):
    opData = np.genfromtxt(opPath,delimiter=',')
    return opData

#Normalize the x(i) values
def normalizeX(x):
    x_mean = np.mean(x)
    x_std_dev = np.std(x)
    
    for i in range(x.shape[0]):
        x[i]= (x[i]-x_mean)/x_std_dev
        
    return x

#Calculating cost function value i.e J(theta)
def calculateCost(X,Y,theta,m):
    return (1 / (2*m)) * np.sum((Y - np.dot(X, theta)) ** 2)

#Performing Gradient Descent to return delta(J(theta))
def grad_descent(X,Y,theta):
    deltaTheta = np.dot(X,theta) - Y
    tmp = np.zeros((2,1))
    for i in range(X.shape[0]):
        tmp[0]+= deltaTheta[i]*X[i][0]
        tmp[1]+= deltaTheta[i]*X[i][1]
    return tmp

#Function for plotting 3d mesh
def fMesh(num, data, line):
    line.set_data(data[:2, :num])
    line.set_3d_properties(data[2, :num])
    return line

#Function for plotting contour
def fContour(num, data, line):
    line.set_data(data[:2, :num])
    return line

#Function for getting J(theta) coordinates
def costFunc(theta_0, theta_1, x1, y1):
    J_mat = 0
    for i in range(m):
        J_mat+= ((theta_0 + theta_1*x1[i]) - y1[i])**2
    J = J_mat/(2*m)
    return J


if __name__ == '__main__':
    ipPath = 'linearX.csv'#input("Enter the input Path:")
    opPath = 'linearY.csv'#input("Enter the output Path:")
    x = load_input(ipPath)
    y = load_output(opPath)

    m = x.shape[0]
        
    x = normalizeX(x)
    
    X = x.reshape(m,1)
    Y = y.reshape(m,1)
    
    X = np.append(np.ones((m, 1)), X, axis = 1)
    
    theta = np.zeros((X.shape[1],1))
    
    #Code for Q1.Part(a)
    eta = 0.1
    Jtheta_old = 0
    Jtheta = calculateCost(X,Y,theta,m)
    t0 = np.array([])
    t1 = np.array([])
    j = np.array([])
    steps = 0
    
    #Gradient Descent Algorithm
    while abs(Jtheta_old - Jtheta) >= 1e-14:
        Jtheta_old = Jtheta
        theta-= (eta * grad_descent(X,Y,theta))/m
        t0 = np.append(t0,theta[0])
        t1 = np.append(t1,theta[1])
        Jtheta = calculateCost(X,Y,theta,m)
        j = np.append(j,Jtheta)
        steps+= 1
    
    print(theta)
    hox = np.dot(X,theta)


#Code for Q1.Part(b)
plt.figure(figsize=(6,4))
plt.plot(X[:,1], hox,'-r', label='Hypothesis')
plt.scatter(X[:,1], Y, s = 15, label='Function',color = 'green')
plt.xlabel('X (Acidity of wine)')
plt.ylabel('Y (Density)')
plt.legend()


#Code for Q1.Part(c)
T0 = t0.reshape(1,t0.T.shape[0])
T1 = t1.reshape(1,t1.T.shape[0])
jt = j.reshape(1,j.T.shape[0])

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')
theta_0 = np.linspace(0, 2, 300)
theta_1 = np.linspace(-1, 1, 300)
cX, cY = np.meshgrid(theta_0, theta_1)

ax.plot_surface(cX, cY, costFunc(cX, cY, X[:,1:], Y), alpha = 0.7)
ax.set_xlabel('Theta_0')
ax.set_ylabel('Theta_1')
ax.set_zlabel('J(theta)')

data = np.append(T0,np.append(T1,jt,axis=0),axis=0)
line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'ro', markersize=8, label='Gradient Descent Movement')
an = animation.FuncAnimation(fig, fMesh, frames=j.shape[0], fargs=(data, line), interval=200)

ax.legend(loc='upper left')
plt.show(block=True)


#Code for Q1.Part(d)
contour = plt.figure(figsize=(8,5))
ax = plt.axes()
con = plt.contour(cX, cY, costFunc(cX, cY, X[:,1:], Y))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.clabel(con)
line, = ax.plot(data[0, 0:1], data[1, 0:1], 'ro', markersize=7, label='Gradient Descent Movement')
ani = animation.FuncAnimation(contour, fContour, frames=j.shape[0], fargs=(data, line), interval=200)
ax.legend(loc='upper left')
plt.show()

