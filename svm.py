#!/usr/bin/python
## Logistic regression for 6.867 hw2

from math import log, exp
import numpy
import scipy.optimize
import cvxopt

from logreg import make_phi

## primal:
## input x, y, C
## TODO get data -- convert to classes
x,y = data
C = 0
def primal(x,y,C):
    ## make phi
    phi = make_phi(x)

    ## primal quad prog.
    ## let n be the number of data points, and let m be the number of features
    n,m = phi.shape
    ## z is [w, b, eps].T
    z = numpy.array([0]*m + [0] + [0]*n)
    Q = numpy.zeros((m+n,m+n))
    for i in range(m):
        Q[i,i] = 1
    c = numpy.vstack(numpy.zeros((m+1,1)), C*numpy.ones((n,1)))
    A = numpy.zeros((n, m+1+n))
    A[:,0:m] = y*phi
    A[:,n] = 1
    A[:,m+1:m+1+n]  = numpy.eye(n)
    A = -A
    g = -numpy.ones((n,1))
    E = d = 0
    ## set up cvxopt
    sol = cvxopt.solvers.qp(Q, c, A, g)
    return sol

## the dual takes a kernel as well
def dual(x,y,C,k):
    ## TODO finish this
    ## make phi
    phi = make_phi(x)

    ## primal quad prog.
    ## let n be the number of data points, and let m be the number of features
    n,m = phi.shape
    ## z is [w, b, eps].T
    z = numpy.array([0]*m + [0] + [0]*n)
    Q = numpy.zeros((m+n,m+n))
    for i in range(m):
        Q[i,i] = 1
    c = numpy.vstack(numpy.zeros((m+1,1)), C*numpy.ones((n,1)))
    A = numpy.zeros((n, m+1+n))
    A[:,0:m] = y*phi
    A[:,n] = 1
    A[:,m+1:m+1+n]  = numpy.eye(n)
    A = -A
    g = -numpy.ones((n,1))
    E = d = 0
    ## set up cvxopt
    sol = cvxopt.solvers.qp(Q, c, A, g)
    return sol

