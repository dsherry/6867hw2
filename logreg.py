#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy

## transform X (Nx1) into feature space (NxM)
def make_phi_old(X, M):
    assert M >= 0, 'Invalid value "%s" for M' %(str(M))
    ## [X**0, X**1, X**2, ... X**M]
    return numpy.hstack([X**i for i in range(M+1)])

"""Enter the feature space via a second order set of basis functions"""
def make_phi(X, M):
    assert M >= 0, 'Invalid value "%s" for M' %(str(M))
    n,D = X.shape
    ## [X**0]
    ones = numpy.ones(n).reshape((n,1))
    ## [X**1] = X (all set)
    ## [xi*xj for i from 0 to D, j from i to D (to avoid duplicates)]
    multinomial = numpy.array([(X[:,i]*X[:,j]).reshape((n,1)) for i in range(D) for j in range(i,D)]).T[0,:,:]
    #print ones.shape, X.shape, multinomial.shape
    return numpy.hstack([ones, X, multinomial])

def sigmoid(X):
    denom = 1.0 + e ** (-1.0 * X)
    return 1.0 / denom

#def negLogLikelihood(w, theta, y, lamduh):
#    return 

def compute_cost(theta, X, y):
    '''
    Comput cost for logistic regression
    '''
    #Number of training samples
    theta.shape = (1, 3)
    m = y.size
    h = sigmoid(X.dot(theta.T))
    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1.0 - y.T).dot(log(1.0 - h))))
    return - 1 * J.sum()

def compute_grad(theta, X, y):
    #print theta.shape
    theta.shape = (1, 3)
    grad = zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = (1.0 / m) * sumdelta * - 1
    theta.shape = (3,)
    return  grad
