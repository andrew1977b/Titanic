

__author__ = 'michaelbinger'

import numpy as np
from pylab import *
import random
import sys
#from random import random, randrange
set_printoptions(suppress = True) # makes for nice printing without scientific notation

########################################################################################################################
########################################################################################################################
########################################################################################################################

def featurescale(X):
    """
    Scales the x data (features) to be of somewhat uniform size. This greatly helps with convergence.
    X=(m,n+1) matrix, with m rows (each representing a training data case).
    The n column vectors, where n=number of features, will be rescaled.
    Note the first column is all 1's and should not be scaled.
    """
    n = np.shape(X)[1]-1
    means = np.zeros(n)
    stds = np.zeros(n)
    for col in xrange(1,n+1):
        means[col-1] = np.mean(X[:,col])
        stds[col-1] = np.std(X[:,col])
        X[:,col] = (X[:,col] - means[col-1])/stds[col-1]
    return [X, means, stds]

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

########################################################################################################################
########################################################################################################################
########################################################################################################################


m = 100 # number of training examples

x1 = np.array([random.randint(1,100) for x in xrange(m)])
x2 = np.array([random.randint(1,100) for x in xrange(m)])
x3 = x2**4 #let's throw an unnecessary term in to the hypothesis

xx = np.ones((m,4))
xx[:,1] = x1
xx[:,2] = x2
xx[:,3] = x3

print "This is X:", xx
fs = featurescale(xx)
xx = fs[0]
print xx
print "means and stds:", fs[1], fs[2]

y = np.zeros((m,1))
for i in xrange(np.size(y)):
    y[i] = x1[i]+2*x2[i] + 5*(random.random() - 1/2)

#print "This is y:", type(y), np.shape(y), y

theta = np.zeros((4,1))

from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = Axes3D(fig)
ax.scatter(x1,x2,y)
xlabel("x1")
ylabel("x2")
title("Random data (from y=x1+2*x2 straight line)")
show()


niter = 1000
alpha = .1

#import sys
#sys.exit()

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
