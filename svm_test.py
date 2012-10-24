from numpy import *
from plotBoundary import *
# import your SVM training code
import svm
import numpy
from logreg import make_phi

M=2

# Define the predictSVM(x) function, which uses trained parameters
def makePredictor(w,b,M):
    def predict(x):
        #print x.shape, w.shape
        ## transform into feature space
        phi = make_phi(x,M)
        #print phi.shape, w.shape
        val = phi.dot(w) + b
        return 1 if val > 0 else -1
    return predict

def train():
    # parameters
    name = 'ls'
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data_'+name+'_train.csv')
    # use deep copy here to make cvxopt happy
    X = train[:, 0:2].copy()
    M=2
    phi = make_phi(X,M)
    n,m = phi.shape
    Y = train[:, 2:3].copy()

    # Carry out training, primal and/or dual
    p = svm.primal(phi,Y,0.01)
    w = numpy.array(p['x'][:m])
    b = p['x'][m]

    predictSVM = makePredictor(w,b,M)

    # plot training results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')

def validate():
    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:, 0:2]
    Y = validate[:, 2:3]
    # plot validation results
    plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

def dummy():
    ## a very simple test
    xDummy = numpy.array([[0,0],[1,1],[1,2]])
    assert xDummy.shape == (3,2)
    phiDummy = make_phi(xDummy,M)
    assert phiDummy.shape == (3,6)
    n,m = phiDummy.shape
    yDummy = numpy.array([[-1], [1], [1]])
    assert yDummy.shape == (3,1)

    # Carry out training, primal and/or dual
    C = 10e10
    p = svm.primal(phiDummy,yDummy,C)
    w = numpy.array(p['x'][:m])
    assert w.shape == (1,6)
    b = p['x'][m]
    assert isinstance(b, (int,float))
    dummyPredictor = makePredictor(w,b,M)

    plotDecisionBoundary(xDummy, yDummy, dummyPredictor, [-1, 0, 1], title = 'SVM Validate')

if __name__=='__main__':
    dummy()
    #pl.show()
