#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
from numpy.random import random
import scipy.optimize

## transform X (Nx1) into feature space (NxM)
def make_phi_old(X, M):
    assert M >= 0, 'Invalid value "%s" for M' %(str(M))
    ## [X**0, X**1, X**2, ... X**M]
    return numpy.hstack([X**i for i in range(M+1)])

##Enter the feature space via a second order set of basis functions
def make_phi(X):
    if len(X.shape) == 1:
        X = X.reshape((1,X.shape[0]))
    n,D = X.shape
    ## [X**0]
    ones = numpy.ones(n).reshape((n,1))
    ## [X**1] = X (all set)
    ## [xi*xj for i from 0 to D, j from i to D (to avoid duplicates)]
    multinomial = numpy.array([(X[:,i]*X[:,j]).reshape((n,1)) for i in range(D) for j in range(i,D)]).T[0,:,:]
    #print ones.shape, X.shape, multinomial.shape
    return numpy.hstack([ones, X, multinomial])

##A sigmoid function, where X is a numpy array of any dimension
def sigmoid(X):
    denom = 1.0 + numpy.exp(-1.0 * X)
    return 1.0 / denom

## the negative log likelihood error function (4.90 in Bishop)
## note: turns out this is the wrong one to use... kept here for later experimentation
def negLogLikelihood_490(w, phi, y, lamduh):
    s = sigmoid(w.T.dot(phi.T)).T
    ## regularizer
    reg = lamduh * (w.T.dot(w))
    return -((y.T.dot(numpy.log(s))) + ((1 - y.T).dot(numpy.log(1 - s)))) + reg
    #return phi.T.dot(sigmoid(w.T.dot(phi.T)).T - y) + (lamduh * numpy.abs(w))

## see above -- this is the incorrect function to use
def negLogLikelihoodGradient_490(w, phi, y, lamduh):
    if len(w.shape) == 1:
        w = w.reshape((w.shape[0],1))
    # print phi.shape
    # print w.shape
    # print y.shape
    return (phi.T.dot(sigmoid(w.T.dot(phi.T)).T - y) + (lamduh * numpy.abs(w))).T[0]

## see above -- this is the incorrect function to use
def hessian_490(w, phi, y, lamduh):
    n,phiD = phi.shape
    R = (sigmoid(w.T.dot(phi.T)) * (1 - sigmoid(w.T.dot(phi.T)))) * numpy.eye(n)
    return phi.T.dot(R.dot(phi)) + lamduh

## compute the predicted output given the weights and some input values in feature space
## this is actually the probability the target is 1 given a position in feature space
def yPredicted(w, b, phi):
    return sigmoid(phi.dot(w) + b)

## the correct error function for logistic regression (7.47 in Bishop)
def error(wb, phi, y, lamduh):
    n,m = phi.shape
    w = wb[:m]
    b = wb[m]
    yp = yPredicted(w, b, phi)
    if not yp.shape == (n,1):
        yp = yp.reshape((n,1))
    return sum(numpy.log(1 + numpy.exp(-yp*y))) + lamduh * (w.T.dot(w))


## Iterative reweighted least squares
def irls(w, phi, y, lamduh):
    n = 0
    try:
        while n < 1000:
            n += 1
            w = w - numpy.linalg.inv(hessian(w, phi, y, lamduh)).dot(negLogLikelihoodGradient(w, phi, y, lamduh))
        return w
    except numpy.linalg.LinAlgError:
        print "LinAlgError"
        print w
        print n

# A wrapper for the error functions, to ease input into scipy optimizers
def weightsWrapper(func):
    def wrapper(w, args):
        return func(w, args[0], args[1], args[2])
    return wrapper

## perform logistic regression -- returns optimal weights and other info from the optimizer
## note: transform input into feature space before inputting here
## opt is the scipy optimizer to use, i.e. fmin, fmin_bfgs, etc
def logreg(phi, y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=False, returnAll=False):
    ## transform into second-order feature space
    n,m = phi.shape
    ## initialize weights -- m for w and 1 for b
    wb = numpy.zeros(m+1)
    return opt(error, wb, args=(phi, y, lamduh), full_output=True, disp=int(printInfo), retall=int(returnAll))

if __name__ == '__main__':
    ## first, test make_phi
    x = numpy.random.random((3,2))
    phi = make_phi(x)
    n,m = phi.shape
    assert (n,m) == (3,6)

    ## now test actual data
    gtol = 1e-5
    train = numpy.loadtxt('data/data_ls_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]
    phi = make_phi(X)
    lamduh=0.1
    #print scipy.optimize.fmin(error, w, args=(phi, y, lamduh), maxfun=100000)
    a = logreg(phi, Y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=True)
    ## return values for scipy optimizers fmin and fmin_bfgs, in order of occurence:
    ## final guess
    ## final value
    ## gradient at final value (fmin_bfgs only)
    ## inverse hessian at final value (fmin_bfgs only)
    ## iterations (fmin only)
    ## function evaluations
    ## gradient evaluations
    ## warnflag (1 if maxfunc reached, 2 if maxiter reached)
    ## allvecs, soln at each step (if retall=1)
    print a
