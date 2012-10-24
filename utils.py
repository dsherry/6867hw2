#!/usr/bin/python
## Utilities common to both logistic regression, SVM, and their test files

from math import log, exp
import numpy
from numpy.random import random
import scipy.optimize
import logreg

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

## Define the predict___(x) function, which uses trained parameters
## This works generally for svms and logreg
def makePredictor(w,b,mode='logreg'):
    assert mode in ['logreg','svm'], "Invalid mode \"%s\"" %(mode)
    threshold = 0.5 if mode=='logreg' else 0.0
    def predict(x):
        ## transform into feature space
        phi = make_phi(x)
        n,m = phi.shape
        val = phi.dot(w) + b
        if mode=='logreg': val = sigmoid(val)
        ## convert to 1/0
        val = val > threshold
        ## replace 0s with -1s
        val = ((val.astype(int) - .5)*2).astype(int)
        return val.reshape((n,1))
    return predict


## given an X/Y pair (validation) and a w/b pair (trained weights), compute the predicted error rate
def validationError(X, Y, w, b, mode='logreg'):
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    assert mode in ['logreg','svm'], "Invalid mode \"%s\"" %(mode)
    predictor = makePredictor(w,b,mode=mode)
    wrong = float(sum(predictor(X) == Y))
    if wrong > n/2: wrong = n - wrong
    return wrong / n
