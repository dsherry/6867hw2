#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
import scipy.optimize
import cvxopt, cvxopt.solvers

from logreg import make_phi
numpy.set_printoptions(threshold=numpy.nan)

## primal:
## input x, y, C

def primal(phi,y,C):
    ## primal quad prog.
    ## let n be the number of data points, and let m be the number of features
    n,m = phi.shape
    ## z is [w, b, eps].T
    z = numpy.array([0]*m + [0] + [0]*n)
    Q = numpy.zeros((m+n+1,m+n+1))
    for i in range(m):
        Q[i,i] = 1
    #print Q[0:7,0:7]
    c = numpy.vstack([numpy.zeros((m+1,1)), C*numpy.ones((n,1))])
    #print c[0:10]
    #A = numpy.zeros((n, m+1+n))
    ## second major change here
    A = numpy.zeros((2*n, m+1+n))
    A[:n,0:m] = y*phi
    ## first major change below
    A[:n,m] = 1
    A[:n,m] = y.T
    A[:n,m+1:]  = numpy.eye(n)
    A = -A
    ## second major change here
    A[n:,m+1:] = numpy.eye(n)
    #print A[1,:]
    ## second major change here
    #g = -numpy.ones((n,1))
    g = numpy.zeros((2*n,1))
    g[:n] = -1
    # E and d are not used in the primal form
    E = d = 0
    ## convert to array
    print numpy.linalg.matrix_rank(Q), numpy.linalg.matrix_rank(c), numpy.linalg.matrix_rank(A), numpy.linalg.matrix_rank(g)
    #comp = numpy.vstack([Q,A])
    #print comp.shape, numpy.linalg.matrix_rank(comp)
    print "n,m=",n,m
    print "Q:",Q.shape
    print Q
    print "c:",c.shape
    print c
    print "A:",A.shape
    print A
    print "g:",g.shape
    print g
    Q = cvxopt.matrix(Q,Q.shape,'d')
    c = cvxopt.matrix(c,c.shape,'d')
    A = cvxopt.matrix(A,A.shape,'d')
    g = cvxopt.matrix(g,g.shape,'d')
    print "n,m=",n,m
    print "Q:",Q.size
    print Q
    print "c:",c.size
    print c
    print "A:",A.size
    print A
    print "g:",g.size
    print g
    ## set up cvxopt
    #sol = cvxopt.solvers.qp(Q, c, A, g)
    sol = cvxopt.solvers.qp(Q, c, A, g)
    return sol

## the dual takes a kernel as well
def dual(phi,y,C,K):
    ## TODO finish this
    pass

if __name__=='__main__':
    # ## test out training
    # name='ls'
    # train = numpy.loadtxt('data/data_'+name+'_train.csv')
    # x = train[:, 0:2].copy()
    # M = 2
    # phi = make_phi(x,M)
    # y = train[:, 2:3].copy()
    # C = .01

    # p=primal(phi,y,C)
    # print p

    ## a very simple training test
    numpy.set_printoptions(threshold=numpy.nan)
    M = 2
    xDummy = numpy.array([[1,0],[-1,0]])
    assert xDummy.shape == (2,2)
    phiDummy = make_phi(xDummy,M)
    assert phiDummy.shape == (2,6)
    print phiDummy
    #n,m = phiDummy.shape
    n,m = xDummy.shape
    yDummy = numpy.array([[1], [-1]])
    assert yDummy.shape == (2,1)

    # Carry out training, primal and/or dual
    C = 0.25
    #C = 0
    #p = primal(phiDummy,yDummy,C)
    p2 = primal(xDummy,yDummy,C)
    w = numpy.array(p2['x'][:m])
#    assert w.shape == (6,1)
    b = p2['x'][m]
