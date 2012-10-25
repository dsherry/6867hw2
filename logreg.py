#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
from numpy.random import random
import scipy.optimize

#from utils import sigmoid
from utils import *
import utils

## compute the predicted output given the weights and some input values in feature space
## this is actually the probability the target is 1 given a position in feature space
def yPredicted(w, b, phi):
    return utils.sigmoid(phi.dot(w) + b)

## the correct error function for logistic regression (7.47 in Bishop)
def error(wb, phi, y, lamduh):
    n,m = phi.shape
    w = wb[:m]
    b = wb[m]
    yp = yPredicted(w, b, phi)
    if not yp.shape == (n,1):
        yp = yp.reshape((n,1))
    return sum(numpy.log(1 + numpy.exp(-yp*y))) + lamduh * (w.T.dot(w))

## perform logistic regression -- returns optimal weights and other info from the optimizer
## note: transform input into feature space before inputting here
## opt is the scipy optimizer to use, i.e. fmin, fmin_bfgs, etc
## basis specifies either a linear or quadratic set of basis functions
def logreg(phi, y, lamduh, opt=scipy.optimize.fmin_bfgs, basis='lin', printInfo=False, returnAll=False):
    ## transform into second-order feature space
    n,m = phi.shape
    ## initialize weights -- m for w and 1 for b
    wb = numpy.zeros(m+1)
    return opt(error, wb, args=(phi, y, lamduh), full_output=True, disp=int(printInfo), retall=int(returnAll))

if __name__ == '__main__':
    ## first, test make_phi
    ## quadratic basis
    M = 2
    x = numpy.random.random((3,2))
    phi = makePhi(x,M)
    n,m = phi.shape
    assert (n,m) == (3,6)

    ## now test actual data
    gtol = 1e-5
    train = numpy.loadtxt('data/data_ls_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]
    phi = makePhi(X, M)
    lamduh=0.1
    #print scipy.optimize.fmin(error, w, args=(phi, y, lamduh), maxfun=100000)
    a = logreg(phi, Y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=True)
    print a
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
