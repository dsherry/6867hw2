#!/usr/bin/python
## Utilities common to both logistic regression, SVM, and their test files

from numpy import *
from scipy import *
from math import log, exp
import numpy
from numpy.random import random
import scipy.optimize
import logreg

from plotBoundary import plotDecisionBoundary

## transform X (NxM) into a simple linear feature space (NxM+1)
def makePhiLinear(X):
    ## [X**0, X**1, X**2, ... X**M]
    if len(X.shape) == 1:
        X = X.reshape((1,X.shape[0]))
    n,m = X.shape
    return X
    #return numpy.hstack([numpy.ones((n,1)),X])

##Enter the feature space via a second order set of basis functions
def makePhiQuadratic(X):
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

def makePhi(X, M):
    if M == 1: return makePhiLinear(X)
    elif M == 2: return makePhiQuadratic(X)
    else: raise Exception('Value %s for M in makePhi must be either 1 or 2'%(M))

##A sigmoid function, where X is a numpy array of any dimension
def sigmoid(X):
    denom = 1.0 + numpy.exp(-1.0 * X)
    return 1.0 / denom

## a predictor for dual form svms with kernels
## from www.support-vector.net/icml-tutorial.pdf
## note: HIGHLY inefficient... redo with numpy
def makeKernelPredictor(w, b, orderM, alpha, xOrig, yOrig, K, C):
    n = len(yOrig)
    def kernelPredictor(xNew):
        ## transform xes into feature space
        phiOrig = makePhi(xOrig, orderM)
        phiNew = makePhi(xNew, orderM)
        #assert phiNew.shape == phiOrig.shape, "phiNew.shape: %s, phiOrig.shape: %s, xOrig.shape: %s %s %s" %(phiNew.shape, phiOrig.shape, xOrig.shape, phiOrig[0,:], xOrig[0,:])
        ## compute w*phi(phiNew) = sum(alphi_i * yOrig_i * K(phiNew,phiOrig_i)
        wphi = sum([alpha[i] * yOrig[i] * K(phiNew.T, phiOrig[i,:].T) for i in xrange(n)])

        ## count the number of data points for which 0 < alpha < C
        mIndices = [i for i in xrange(n) if 0 < alpha[i] < C]
        M = len(mIndices)
        ## average the calculated b values
        b = sum([yOrig[k] - sum([alpha[i] * yOrig[i] * K(phiOrig[k,:].T,phiOrig[i,:].T) for i in xrange(n)]) for k in mIndices])/float(M)
        #b = sum([yOrig[j] - sum([alpha[i] * yOrig[i] * K(phiOrig[i,:],phiOrig[j,:]) for i in sIndices]) for j in mIndices])/float(M)
        return 1 if wphi + b > 0 else -1

    return kernelPredictor

## Define the predict___(x) function, which uses trained parameters
## This works generally for svms and logreg
## makePhi is the basis function used to transform into feature space
def makePredictor(w,b,M,mode='lr',kernel=None):
    assert mode in ['lr','svm'], "Invalid mode \"%s\"" %(mode)
    threshold = 0.5 if mode=='lr' else 0.0
    def predict(x):
        ## transform into feature space
        try:
            phi = makePhi(x,M)
        except ValueError:
            print x
            print x.shape,M
            assert False, "ValueError in utils.makePredictor"
        n,m = phi.shape
        val = phi.dot(w) + b
        if mode=='lr': val = sigmoid(val)
        ## convert to 1/0
        val = val > threshold
        ## replace 0s with -1s
        val = ((val.astype(int) - .5)*2).astype(int)
        return val.reshape((n,1))
    return predict


## given an X/Y pair and a w/b pair (trained weights), compute the predicted error rate
def getError(X, Y, w, b, M, mode='lr', predictor=None):
    n = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert Y.shape[1] == 1
    assert mode in ['lr','svm'], "Invalid mode \"%s\"" %(mode)
    if not predictor:
        predictor = makePredictor(w,b,M,mode=mode)
    wrong = float(sum([predictor(X[i,:]) == Y[i] for i in xrange(n)]))
    if wrong > n/2: wrong = n - wrong
    return wrong / n

class Train:
    def __init__(self, params, problemClass='svm',basisfunc='lin', datapath='data/data_%s_%s.csv', dataSetName='ls',plot=False, printInfo=False):
        assert isinstance(params, dict)
        assert 'lamduh' in params or 'C' in params
        self.params = params
        self.dataSetName = dataSetName
        self.datapath = datapath
        self.plot = plot
        self.printInfo = printInfo
        ## handle problem class
        if not problemClass.lower() in ['svm', 'lr']:
            raise Exception('Value "%s" for problemClass must be either "svm" or "lr"'%(problemClass))
        self.problemClass = problemClass.upper()
        # handle basis function
        if basisfunc=='lin': self.M=1
        elif basisfunc=='quad': self.M=2
        else: raise Exception('Value "%s" for basisfunc must be either "lin" or "quad"'%(basisfunc))

    def __call__(self):
        return self._computeTVError()

    def _computeError(self, X, Y, name=' (unnamed)'):
        ## make a predictor and get training error
        predictor, tErr = self._trainPredictError(X, Y)

        # plot training results
        if self.plot:
            plotDecisionBoundary(X, Y, predictor, [-1, 0, 1], title = self.problemClass + name)

        return tErr

    ## return the training and validation error
    def _computeTVError(self):
        ## load data
        train = numpy.loadtxt(self.datapath %(self.dataSetName,'train'))
        self.tX = train[:, 0:2].copy()
        #self.tPhi = makePhi(self.tX,self.M)
        #self.n,self.m = self.tPhi.shape
        self.tY = train[:, 2:3].copy()

        ## make a predictor and get training error
        self.predictor, self.tErr = self._trainPredictError(self.tX, self.tY)

        # plot training results
        if self.plot:
            plotDecisionBoundary(self.tX, self.tY, self.predictor, [-1, 0, 1], title = self.problemClass + ' Train')

        ## load validation data
        validate = numpy.loadtxt(self.datapath %(self.dataSetName,'validate'))
        self.vX = validate[:, 0:2].copy()
        self.vY = validate[:, 2:3].copy() ## actually a width of 1 for this data

        # plot validation results
        if self.plot:
            plotDecisionBoundary(self.vX, self.vY, self.predictor, [-1, 0, 1], title = self.problemClass + ' Validate')

        # print validation error
        self.vErr = self._getError(self.vX, self.vY, self.predictor)
        return self.tErr, self.vErr

    ## make a predictor appropriate for the problem class, using trained weights
    def _getPredictor(self):
        # if self.problemClass.lower() == 'svm':
        #     return makeKernelPredictor(self.w, self.b, self.alphaD, self.tX, self.tY, self.K)
        return makePredictor(self.w,self.b,self.M,mode=self.problemClass.lower())

    ## compute the error, given X and Y
    def _getError(self, X, Y, predictor):
        return getError(X, Y, self.w, self.b, self.M, mode=self.problemClass.lower(), predictor=self.predictor)

    ## train and return a predictor and the training error
    def _trainPredictError(self, X, Y):
        # given training params, get w and b
        self.w, self.b = self._train(X, Y)
        # make predictor
        self.predictor = self._getPredictor()
        # get training error
        tErr = self._getError(X, Y, self.predictor)
        return self.predictor, tErr

    ## given problem-specific params (i.e., C or lambda), return weights w and b
    ## this is the one function which must be implemented by subclasses
    def _train(self, X, Y):
        raise NotImplemented('This should be set by a subclass')


## plots training and validation errors vs lambda (or C)
## lambdaC, tError and vError have same dims.
## lambdaC can be either lambda (LR) or C (SVM)
## extra is used by svm_test to add in info about the kernel
def plotTVError(lambdaC, tError, vError, problemClass='lr', varName='$\lambda$', linQuad='linear', extra=''):
    import pylab as pl
    fig = pl.figure()
    pl.plot(lambdaC, tError, 'b', lambdaC, vError, 'g')
    fig.gca().set_xscale('log', basex=10)
    pl.title('%s Error v.s. %s with %s basis functions%s' %(problemClass.upper(), varName, linQuad, extra))
    pl.xlabel('%s, $log_{10}$ scale' %(varName))
    pl.ylabel('Error')
    pl.legend(('Training error', 'Validation error'), loc=2)

