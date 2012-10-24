from numpy import *
from plotBoundary import *
# import your LR training code
import scipy

import logreg
from utils import *

def train(name='ls', lamduh=0.1):
    # parameters
    name = 'ls'
    print '======Training======'
    # load data from csv files
    train = loadtxt('data/data_'+name+'_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]

    # Carry out training.
    phi = make_phi(X)
    n,m = phi.shape
    result = logreg.logreg(phi, Y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=True)
    w = result[0][:m]
    b = result[0][m]

    # Define the predictLR(x) function, which uses trained parameters
    predictLR = makePredictor(w,b,mode='logreg')

    # plot training results
    plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:,0:2]
    Y = validate[:,2:3]

    # plot validation results
    plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')

    # print validation error
    err = validationError(X, Y, w, b, mode='logreg')
    print err

if __name__ == '__main__':
    train(lamduh=0)
    train(lamduh=0.01)
    train(lamduh=0.1)
    train(lamduh=1)
    train(lamduh=10)
