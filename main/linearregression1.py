

__author__ = 'michaelbinger'

import numpy as np
from random import random, randrange
from pylab import *
import matplotlib
#import random


m = 100 # number of training data examples

x = np.array([randint(1,100) for x in xrange(m)])
y = np.zeros((m,1))

for i in xrange(m):
    r = random() #random number btw 0 and 1
    y[i] = x[i] + 100*(r - 0.5) #The true distribution of y is simply y=x, but we add some randomness to simulate real data

scatter(x,y)
xlabel("x")
ylabel("y")
title("Random data (from y=x straight line)")
show()

print "features m = %i" %m

xx = np.ones((m,2))
xx[:,1] = x
print "This is X"
print xx
print "This is y"
print y

theta = np.zeros((2,1))
print "This is theta"
print theta

niter = 100
alpha = 0.0001

def Jcost(X,y,theta):
    """
    cost function J for linear regression of one variable
    """
    m = np.size(y)
    h = np.dot(X,theta) # hypothesis function
    sqErrors = (h - y) ** 2
    J = (1.0 / (2 * m)) * sqErrors.sum()
    return J

def gradientdescent(X,y,theta,alpha,niter):
    """
    performs gradient descent algorithm for linear regression
    """
    m = np.size(y) # # of features
    Jsteps = np.zeros((niter,1))
    for i in xrange(niter):
        h = np.dot(X,theta)
        err0 = np.dot((h-y).T, X[:, 0])
        err1 = np.dot((h-y).T, X[:, 1])
        theta[0] = theta[0] - (alpha / m) * err0.sum()
        theta[1] = theta[1] - (alpha / m) * err1.sum()

        Jsteps[i, 0] = Jcost(X, y, theta)
    return [theta,Jsteps.T]

graddes = gradientdescent(xx,y,theta,alpha,niter)
thetapred = graddes[0]
Jsteps = graddes[1]
print "theta=", thetapred
#Figure out how to write this as
# print "theta = %??" %theta
# where ?? is unknown. Or something simple like that.


#print Jsteps #uncomment to see the value of J at each step


scatter(x,y,c='r')
plot(x,x,c='b')
plot(x,thetapred[0]+thetapred[1]*x,c='g')
xlabel("x")
ylabel("y")
title("LR pred in green, 'true' distribution in blue")
show()

scatter(np.arange(niter)+1,Jsteps)
xlabel("Number of iterations")
ylabel("Jcost")
title("The convergence of the cost function")
show()
