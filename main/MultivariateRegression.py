

__author__ = 'michaelbinger'

import numpy as np
from pylab import *
import random
#from random import random, randrange
set_printoptions(suppress = True) # makes for nice printing without scientific notation

npts = 100

x1 = np.array([random.randint(1,100) for x in xrange(npts)])
x2 = np.array([random.randint(1,100) for x in xrange(npts)])
x3 = x2**4

m = np.size(x1)
print "features m = %i" %m

xx = np.ones((m,4))
xx[:,1] = x1
xx[:,2] = x2
xx[:,3] = x3

print "This is X"
print xx

y = np.zeros((npts,1))
for i in xrange(np.size(y)):
    y[i] = x1[i]+2*x2[i] #+ 5*(random.random() - 1/2)


print "This is y"
print type(y), np.shape(y), y

#Feature scaling
x1s = ( x1-np.mean(x1) ) / np.std(x1)
x2s = ( x2-np.mean(x2) ) / np.std(x2)
x3s = ( x3-np.mean(x3) ) / np.std(x3)

xx[:,1] = x1s
xx[:,2] = x2s
xx[:,3] = x3s
print "This is X with feature scaling"
print xx


from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y)
xlabel("x1")
ylabel("x2")
title("Random data (from y=x1+2*x2 straight line)")
show()

theta = np.zeros((4,1))
print "This is theta"
print theta

niter = 1000
alpha = .1

#import sys
#sys.exit()

def Jcost(X,y,theta):
    """
    Cost function J for linear regression of one variable.
    X is (m,n+1) the feature matrix, where m = number of data examples and n = number of features
    y is the dependent variable we are trying to train on. y is a (m,1) column vector
    theta is a (n+1,1) column vector
    """
    m = np.size(y) # number of features
    h = np.dot(X,theta) # hypothesis function is a (m,1) column vector
    sqErrors = (h - y) ** 2 #squared element-wise, still (m,1) vector
    J = (1.0 / (2 * m)) * sqErrors.sum() # sum up the sqErrors for each term
    return J

def gradientdescent(X,y,theta,alpha,niter):
    """
    Performs gradient descent algorithm for linear regression.
    X is (m,n+1) the feature matrix, where m = number of data examples and n = number of features
    y is the dependent variable we are trying to train on. y is a (m,1) column vector
    theta is a (n+1,1) column vector
    alpha is the learning rate.
    niter is the number of iterations.
    """
    m = np.size(y) # # of features
    Jsteps = np.zeros((niter,1))
    for i in xrange(niter):
        h = np.dot(X,theta) #(m,1) column vector
        err_x = np.dot((h - y).T, X) #(1,n+1) row vector
        theta = theta - (alpha / m) * err_x.T #(n+1,1) column vector
        Jsteps[i, 0] = Jcost(X, y, theta)
    return [theta.T, Jsteps.T]

graddes = gradientdescent(xx,y,theta,alpha,niter)

thetapred = graddes[0]
Jsteps = graddes[1]
print thetapred
#print Jsteps


scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()
