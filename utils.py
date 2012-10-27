#!/usr/bin/python
## Utilities common to both logistic regression, SVM, and their test files

from numpy import *
from scipy import *
from math import log, exp
import numpy
from numpy.random import random
import scipy.optimize
import logreg
import time

from plotBoundary import plotDecisionBoundary

## profiling decorator
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called+= 1
        before = time.time()
        retval = fn(*args, **kwargs)
        duration = time.time() - before
        wrapper.avgtime = ((wrapper.avgtime * wrapper.called) + duration) / (wrapper.called + 1)
        return retval
    wrapper.called = 0
    wrapper.avgtime = 0
    wrapper.__name__= fn.__name__
    return wrapper

def counting(other):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            other.called= 0
            other.avgTime = 0
            try:
                return fn(*args, **kwargs)
            finally:
                print '%s was called %i times, avg time of %s millisec' % (other.__name__, other.called, other.avgtime*10e3)
        wrapper.__name__= fn.__name__
        return wrapper
    return decorator

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        diff = time.time() - ts

        #print '%r (%r, %r) %f sec' % \
        #      (method.__name__, args, kw, te-ts)
        print '%r %f millisec' % \
              (method.__name__, diff*10e3)
        return result
    return timed

## test the profiling decorator
# @counted
# def foo():
#     print 'baz'

# @counting(foo)
# def bar():
#     foo()
#     foo()
#     foo()

## transform X (NxM) into a simple linear feature space (NxM+1)
@counted
def makePhiLinear(X):
    ## [X**0, X**1, X**2, ... X**M]
    if len(X.shape) == 1:
        #assert False
        X = X.reshape((1,X.shape[0]))
    n,m = X.shape
    return X
    #return numpy.hstack([numpy.ones((n,1)),X])

##Enter the feature space via a second order set of basis functions
@counted
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

@counted
def makePhi(X, M):
    if M == 1: return makePhiLinear(X)
    elif M == 2: return makePhiQuadratic(X)
    else: raise Exception('Value %s for M in makePhi must be either 1 or 2'%(M))

##A sigmoid function, where X is a numpy array of any dimension
@counted
def sigmoid(X):
    denom = 1.0 + numpy.exp(-1.0 * X)
    return 1.0 / denom

## a predictor for dual form svms with kernels
## from www.support-vector.net/icml-tutorial.pdf
## note: HIGHLY inefficient... redo with numpy
@counted
def makeKernelPredictor(w, b, orderM, alpha, xOrig, yOrig, K, C, params=None):
    n = len(yOrig)
    ## transform xes into feature space
    phiOrig = makePhi(xOrig, orderM)
    ## count the number of data points for which 0 < alpha < C
    mIndices = [i for i in xrange(n) if 0 < alpha[i] < C]
    M = len(mIndices)
    ## average the calculated b values
    before = time.time()
    print "entering b"
    KM = numpy.zeros((n,1))
    b = 0
    for k in mIndices:
        for i in xrange(n):
            KM[i] = K(phiOrig[k,:].T, phiOrig[i,:].T)
        b += yOrig[k] - sum(alpha * yOrig * KM)
    if M == 0: b = 0
    else: b = b / float(M)
    #b = sum([yOrig[k] - sum([alpha[i] * yOrig[i] * K(phiOrig[k,:].T,phiOrig[i,:].T) for i in xrange(n)]) for k in mIndices])/float(M)
    print time.time()-before
    print 'done'
    beta = params['beta']
    print "beta = " + str(beta)
    kernelName = params['kernelName']
    @counted
    def kernelPredictor(xNew):
        ## transform xes into feature space
        phiNew = makePhi(xNew, orderM)
        ## compute w*phi(phiNew) = sum(alphi_i * yOrig_i * K(phiNew,phiOrig_i)
        # KM = numpy.array([K(phiNew.T, phiOrig[i,:].T) for i in xrange(n)])
        # for i in xrange(n):
        #     KM[i] = K(phiNew.T, phiOrig[i,:].T)

        if kernelName == 'Gaussian':
            d = phiNew - phiOrig
            ds = numpy.sum(d * d, axis=1)
            KM = numpy.exp(-beta * (ds))
        elif kernelName == 'Second-order polynomial':
            d = 1 + numpy.sum(phiNew * phiOrig, axis=1)
            KM = numpy.sum(d * d, axis=1)
        else:
            KM = numpy.sum(phiNew * phiOrig, axis=1)

        assert KM.shape == (n,), KM.shape
        assert alpha.T.shape == (1,n), alpha.shape
        assert yOrig.T.shape == (1,n), yOrig.shape
        wphi = sum(alpha.T * yOrig.T * KM)
        return 1 if wphi > -b else -1

    print type(b), b
    print "returning kernelPredictor"
    return kernelPredictor

## Define the predict___(x) function, which uses trained parameters
## This works generally for svms and logreg
## makePhi is the basis function used to transform into feature space
#@counted
def makePredictor(w,b,M,mode='lr',kernel=None):
    assert mode in ['lr','svm'], "Invalid mode \"%s\"" %(mode)
    threshold = 0.5 if mode=='lr' else 0.0
    @counted
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

class Train(object):
    def __init__(self, params, problemClass='svm',basisfunc='lin', datapath='data/data_%s_%s.csv', dataSetName='ls',plot=False, printInfo=False, extra="", meshSize=200.):
        assert isinstance(params, dict)
        assert 'lamduh' in params or 'C' in params
        self.extra = extra
        self.params = params
        self.dataSetName = dataSetName
        self.datapath = datapath
        self.meshSize = float(meshSize)
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
        self.predictor, self.tErr = self._trainPredictError(X, Y)

        # plot training results
        if self.plot:
            suffix = self._generateTitle() % ((str(self.tErr), self.params['lamduh']) if self.problemClass.lower() == 'lr' else (self.tErr, 'primal' if self.params['primal'] else 'dual',self.params['C']))
            title = self.problemClass + " Train" + suffix
            if self.problemClass.lower() == 'svm':
                if not self.params['primal']:
                    title += ", " + self.params['kernelName'] + " kernel"
            plotDecisionBoundary(X, Y, self.predictor, [-1, 0, 1], title = title, meshsize=self.meshSize)
        return self.tErr

    ## generate a plot title suffix
    def _generateTitle(self):
        return "" ## override in subclasses

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
            suffix = self._generateTitle() % ((str(self.tErr), self.params['lamduh']) if self.problemClass.lower() == 'lr' else (self.tErr, 'primal' if self.params['primal'] else 'dual',self.params['C']))
            title = self.problemClass + " Train" + suffix
            if self.problemClass.lower() == 'svm':
                if not self.params['primal']:
                    title += ", " + self.params['kernelName'] + " kernel"
            plotDecisionBoundary(self.tX, self.tY, self.predictor, [-1, 0, 1], title = title, meshsize = self.meshSize)

        ## load validation data
        validate = numpy.loadtxt(self.datapath %(self.dataSetName,'validate'))
        self.vX = validate[:, 0:2].copy()
        self.vY = validate[:, 2:3].copy() ## actually a width of 1 for this data

        # print validation error
        self.vErr = self._getError(self.vX, self.vY, self.predictor)

        # plot validation results
        if self.plot:
            suffix = self._generateTitle() % ((str(self.vErr), self.params['lamduh']) if self.problemClass.lower() == 'lr' else (self.vErr, 'primal' if self.params['primal'] else 'dual',self.params['C']))
            title = self.problemClass + " Validate" + suffix
            if self.problemClass.lower() == 'svm':
                if not self.params['primal']:
                    title += ", " + self.params['kernelName'] + " kernel"
            plotDecisionBoundary(self.vX, self.vY, self.predictor, [-1, 0, 1], title = title, meshsize = self.meshSize)

        ## compute the geometric margin
        gm = 1.0 / numpy.linalg.norm(self.w)
        if self.problemClass.lower() == 'lr':
            return self.tErr, self.vErr, gm
        elif self.params['primal']:
            ## calculate the number of support vectors
            self.sv = self.numSupport(self.slack)
        return self.tErr, self.vErr, gm, self.sv

    ## return the number of points that are significantly higher than 0 (i.e., > 10-e7)
    @counted
    def numSupport(self, vec):
        return sum(vec > 10e-6)

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
        # s = time.time()
        # print self.predictor(numpy.array([100,100]))
        # e = time.time() - s
        # print e
        # assert False

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
def plotTVError(lambdaC, tError, vError, gm=None, sv=None, problemClass='lr', varName='$\lambda$', linQuad='linear', extra=''):
    import pylab as pl
    fig = pl.figure()
    fig.gca().set_xscale('log', basex=10)
    ## TODO add here.
    host = fig.add_subplot(111)
    par1 = host.twinx()
    #par1.set_yscale('log', basex=10)

    host.set_xlabel('%s, $log_{10}$ scale' %(varName))
    #host.set_title('%s Error v.s. %s with %s basis functions%s' %(problemClass.upper(), varName, linQuad, extra))
    host.set_title('%s Error v.s. %s%s' %(problemClass.upper(), varName, extra))

    p1 = host.plot(lambdaC, tError, 'b', label = "Train")
    host.set_ylim(0,0.5)
    p2 = host.plot(lambdaC, vError, 'g', label="Validate")
    #print gm
    p3 = par1.plot(lambdaC, gm, 'r', label="Geometric margin")
    #par1.set_ylim(0,max(gm[1:])+.01)
    par1.set_ylim(0,max(gm)+.01)

    host.set_ylabel("Error")
    par1.set_ylabel("Distance")

    lines = p1 + p2 + p3

    #if not gm == None:
    #    pl.plot(lambdaC, gm, 'r')
    if not sv == None:
        print "sv"
        fig.subplots_adjust(right=0.75)
        par2 = host.twinx()
        par2.spines['right'].set_position(('axes',1.2))

        par2.set_frame_on(True)
        par2.patch.set_visible(False)
        for sp in par2.spines.itervalues():
            sp.set_visible(False)

        par2.spines['right'].set_visible(True)

        p4 = par2.plot(lambdaC, sv, 'ko', label="Support vectors")
        lines.append(p4[0])
        par2.set_ylim(0,max(sv)+1)
        par2.set_ylabel("Count")

    host.legend(lines, [l.get_label() for l in lines], loc=1)

    #fig.gca().set_xscale('log', basex=10)
    #pl.title('%s Error v.s. %s with %s basis functions%s' %(problemClass.upper(), varName, linQuad, extra))
    #pl.xlabel('%s, $log_{10}$ scale' %(varName))
    #pl.ylabel('Error')
    #pl.legend(('Training error', 'Validation error'), loc=2)

