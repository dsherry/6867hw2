from numpy import *
from plotBoundary import *
# import your SVM training code
import svm
import numpy
from utils import *
from svm import *

class SVMTrain(Train):
    def _train(self, X, Y):
        ## first, cvxopt output suppression
        if not self.printInfo: cvxopt.solvers.options['show_progress'] = False
        phi = makePhi(X,self.M)
        n,m = phi.shape
        primal = self.params['primal']
        C = self.params['C']
        if primal:
            self.result = svm.primal(phi,Y,C)
            w = numpy.array(self.result['x'][:m])
            b = self.result['x'][m]
        else:
            kernel = self.params['kernel']
            self.result = svm.dual(phi,Y,C)

            d = dual(phiDummy,yDummy, C, kernel)
            self.alphaD = numpy.array(d['x'])
            w,b,self.S,self.M = svm.dualWeights(phiDummy, yDummy, kernel, self.alphaD, C)
        return w,b

def dummy():
    ## a very simple test
    xDummy = numpy.array([[0,0],[1,1],[1,2]])
    assert xDummy.shape == (3,2)
    ## quadratic basis function
    M=2
    phiDummy = makePhi(xDummy,M)
    assert phiDummy.shape == (3,6)
    n,m = phiDummy.shape
    yDummy = numpy.array([[-1], [1], [1]])
    assert yDummy.shape == (3,1)

    # Carry out training, primal and/or dual
    C = 10e10
    C = 0.25
    p = svm.primal(phiDummy,yDummy,C)
    w = numpy.array(p['x'][:m])
    assert w.shape == (6,1)
    b = p['x'][m]
    assert isinstance(b, (int,float))
    dummyPredictor = makePredictor(w,b,M,'svm')

    plotDecisionBoundary(xDummy, yDummy, dummyPredictor, [-1, 0, 1], title = 'SVM Validate')

if __name__=='__main__':
    # SVMTrain('ls',C=0)
    # SVMTrain('ls',C=0.01)
    # SVMTrain('ls',C=0.1)
    # SVMTrain('ls',C=1)
    # SVMTrain('ls',C=10)

    print SVMTrain({'primal':True,'C':0.01}, printInfo=True)()
