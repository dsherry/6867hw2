from numpy import *
from plotBoundary import *
# import your SVM training code
import svm
import numpy
from utils import *
from svm import *

def train(name,C=1):
    # parameters
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data_'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    phi = make_phi(X)
    n,m = phi.shape
    Y = train[:, 2:3].copy()

    # Carry out training, primal and/or dual
    p = svm.primal(phi,Y,C)
    w = numpy.array(p['x'][:m])
    b = p['x'][m]

    # make predictor
    predictSVM = makePredictor(w,b,'svm')

    # plot training results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')

    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:, 0:2]
    Y = validate[:, 2:3]

    # make predictor
    predictSVM = makePredictor(w,b,'svm')

    # plot validation results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

    # print validation error
    err = validationError(X, Y, w, b, mode='svm')
    print err

def dummy():
    ## a very simple test
    xDummy = numpy.array([[0,0],[1,1],[1,2]])
    assert xDummy.shape == (3,2)
    phiDummy = make_phi(xDummy)
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
    dummyPredictor = makePredictor(w,b,'svm')

    plotDecisionBoundary(xDummy, yDummy, dummyPredictor, [-1, 0, 1], title = 'SVM Validate')

if __name__=='__main__':
    train('ls',C=0)
    train('ls',C=0.01)
    train('ls',C=0.1)
    train('ls',C=1)
    train('ls',C=10)
