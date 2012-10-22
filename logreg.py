#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
import scipy.optimize

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

"""A sigmoid function, where X is a numpy array of any dimension"""
def sigmoid(X):
    denom = 1.0 + numpy.exp(-1.0 * X)
    return 1.0 / denom

def negLogLikelihood(w, phi, y, lamduh):
    s = sigmoid(w.T.dot(phi.T)).T
    ## regularizer
    reg = lamduh * (w.T.dot(w))
    return -((y.T.dot(numpy.log(s))) + ((1 - y.T).dot(numpy.log(1 - s)))) + reg
    #return phi.T.dot(sigmoid(w.T.dot(phi.T)).T - y) + (lamduh * numpy.abs(w))

def negLogLikelihoodGradient(w, phi, y, lamduh):
    if len(w.shape) == 1:
        w = w.reshape((w.shape[0],1))
    # print phi.shape
    # print w.shape
    # print y.shape
    return (phi.T.dot(sigmoid(w.T.dot(phi.T)).T - y) + (lamduh * numpy.abs(w))).T[0]

def hessian(w, phi, y, lamduh):
    n,phiD = phi.shape
    R = (sigmoid(w.T.dot(phi.T)) * (1 - sigmoid(w.T.dot(phi.T)))) * numpy.eye(n)
    return phi.T.dot(R.dot(phi)) + lamduh

"""Iterative reweighted least squares"""
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

## test
train = numpy.loadtxt('data/data_ls_train.csv')
X = train[:,0:2]
y = train[:,2:3]
phi = make_phi(X,3)
lamduh=0.1
#w = numpy.zeros(phi.shape[1]).reshape(phi.shape[1],1)
w = numpy.ones(phi.shape[1])

def weightsWrapper(func):
    def wrapper(w, args):
        return func(w, args[0], args[1], args[2])
    return wrapper

#irls(w, phi, y, lamduh)

def fmin_bfgs_logreg(w, phi, y, lamduh):
    #w = scipy.optimize.fmin_bfgs(negLogLikelihood, w, fprime=negLogLikelihoodGradient, args=(phi, y, lamduh))
    w = scipy.optimize.fmin_bfgs(negLogLikelihood, w, args=(phi, y, lamduh))
    return w

gtol = 1e-5
#print scipy.optimize.fmin_bfgs(negLogLikelihood, w, args=(phi, y, lamduh), gtol=gtol)
#print scipy.optimize.fmin_ncg(negLogLikelihood, w, negLogLikelihoodGradient, args=(phi, y, lamduh) )
print w
w = w * 1.9
print scipy.optimize.fmin_bfgs(negLogLikelihood, w, args=(phi, y, lamduh))
#print scipy.optimize.fmin_bfgs(negLogLikelihood, w, fprime=negLogLikelihoodGradient, args=(phi, y, lamduh), gtol=0.1)
#print scipy.optimize.fmin_ncg(negLogLikelihood, w, negLogLikelihoodGradient, args=(phi, y, lamduh))

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
