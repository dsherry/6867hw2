#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
import scipy.optimize
import cvxopt, cvxopt.solvers

from utils import *

numpy.set_printoptions(threshold=numpy.nan)

## primal SVM
def primal(phi,y,C, debug=False):
    ## primal quad prog.
    ## let n be the number of data points, and let m be the number of features
    n,m = phi.shape
    Q = numpy.zeros((m+n+1,m+n+1))
    for i in range(m):
        Q[i,i] = 1
    c = numpy.vstack([numpy.zeros((m+1,1)), C*numpy.ones((n,1))])
    A = numpy.zeros((2*n, m+1+n))
    A[:n,0:m] = y*phi
    A[:n,m] = y.T
    A[:n,m+1:]  = numpy.eye(n)
    A[n:,m+1:] = numpy.eye(n)
    A = -A
    g = numpy.zeros((2*n,1))
    g[:n] = -1
    ## E and d are not used in the primal form
    ## convert to array
    ## have to convert everything to cxvopt matrices
    Q = cvxopt.matrix(Q,Q.shape,'d')
    c = cvxopt.matrix(c,c.shape,'d')
    A = cvxopt.matrix(A,A.shape,'d')
    g = cvxopt.matrix(g,g.shape,'d')
    ## set up cvxopt
    ## z (the vector being minimized for) in this case is [w, b, eps].T
    sol = cvxopt.solvers.qp(Q, c, A, g)
    return sol

## the dual takes a kernel function as well
def dual(phi,y,C,K, debug=False):
    ## primal quad prog.
    ## let n be the number of data points, and let m be the number of features
    n,m = phi.shape
    ## the kernel function takes two 1xm x-vectors and returns a scalar
    ## first, compute the nxn matrix K(x_i, x_j)
    KM = numpy.zeros((n,n))
    for i in range(n):
        for j in range(n):
            KM[i,j] = K(phi[i,:], phi[j,:])
    ## multiply in y_i*y_j
    Q = y.dot(y.T) * KM
    c = -numpy.ones((n,1))
    A = numpy.zeros((2*n, n))
    A[:n,:] = numpy.eye(n)
    A[n:,:] = -numpy.eye(n)
    g = numpy.zeros((2*n,1))
    g[:n] = C
    E = y.T
    d = numpy.array([[0]])
    ## convert to array
    ## have to convert everything to cxvopt matrices
    Q = cvxopt.matrix(Q,Q.shape,'d')
    c = cvxopt.matrix(c,c.shape,'d')
    A = cvxopt.matrix(A,A.shape,'d')
    g = cvxopt.matrix(g,g.shape,'d')
    E = cvxopt.matrix(E,E.shape,'d')
    d = cvxopt.matrix(d,d.shape,'d')
    ## set up cvxopt
    ## z (the vector being minimized for) in this case is [w, b, eps].T
    sol = cvxopt.solvers.qp(Q, c, A, g, E, d)
    return sol

## given the alpha Lagrange coefficients from dual, calculate the weights and offset
def dualWeights(x, y, K, alpha, C):
    assert len(x.shape) > 1
    n,m = x.shape
    assert y.shape == (n,1)
    assert alpha.shape == (n,1)
    ## calculate weights
    w = sum([alpha[i] * y[i] * x[i,:] for i in range(n)])
    ## calculate offset b
    ## get the indices of the support vectors
    sIndices = [i for i in range(n) if alpha[i] > 0]
    S = len(sIndices)
    ## first count the number of data points for which 0 < alpha < C
    mIndices = [i for i in range(n) if 0 < alpha[i] < C]
    M = len(mIndices)
    b = sum([y[j] - sum([alpha[i] * y[i] * K(x[i,:],x[j,:]) for i in sIndices]) for j in mIndices])/float(M)
    return w, b, S, M

## given an array with shape (r) or (r,1), will return one with shape (r,1)
def fixSingleton(x):
    return x.reshape((x.shape[0],1)) if len(x.shape) == 1 else x

def fixSingletons(*args):
    return [fixSingleton(z) for z in args]

## nifty kallable Kernel wrapper klass
class Kernel:
    def __init__(self, K):
        self.K = K
    def __call__(self, a, b):
        ## make sure they're 2d
        a,b = fixSingletons(a,b)
        for x,name in [(a,"a"),(b,"b")]:
            assert x.shape[1] == 1, "Vector %s has shape %s" %(name, str(x.shape))
        return self.K(a,b)

## define a linear kernel function, where inputs a and b are each 1xm input vectors
linearKernel = Kernel(lambda a,b: a.T.dot(b))
squaredKernel = Kernel(lambda a,b: a.T.dot(b)**2)
beta = 0.1
gaussianKernel = Kernel(lambda a,b: exp(-beta*((a-b).T.dot(a-b))))

if __name__=='__main__':
    ## a very simple training test
    numpy.set_printoptions(threshold=numpy.nan)
    M = 2
    xDummy = numpy.array([[1,0],[-1,0]])
    assert xDummy.shape == (2,2)
    phiDummy = makePhi(xDummy,M)
    assert phiDummy.shape == (2,6)
    print phiDummy
    n,m = phiDummy.shape
    #n,m = xDummy.shape
    yDummy = numpy.array([[1], [-1]])
    assert yDummy.shape == (2,1)

    ## Carry out training, primal and/or dual
    C = 0.25
    ## primal
    p = primal(phiDummy,yDummy,C, debug=True)
    wP = numpy.array(p['x'][:m])
    bP = p['x'][m]
    xValidate = numpy.array([[100,0],[-100,231232],[0,123123123]])
    yP = makePhi(xValidate,M).dot(wP) + bP
    print yP>0

    ## dual
    d = dual(phiDummy,yDummy,C, linearKernel, debug=True)
    alphaD = numpy.array(d['x'])
    print alphaD
    w,b,S,M = dualWeights(phiDummy, yDummy, linearKernel, alphaD, C)

