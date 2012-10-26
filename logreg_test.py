from numpy import *
from plotBoundary import *
# import your LR training code
import scipy

import logreg
from logreg import yPredicted
from utils import *

class LRTrain(Train):
    def _train(self, X, Y):
        phi = makePhi(X,self.M)
        n,m = phi.shape
        self.result = logreg.logreg(phi, Y, self.params['lamduh'], opt=scipy.optimize.fmin_bfgs, printInfo=self.printInfo)
        w = self.result[0][:m]
        b = self.result[0][m]
        return w,b

def train(lamduh=0.1, basisfunc='lin', plot=False, optimizePrint=False, name='ls'):
    # handle basis function
    if basisfunc=='lin': M=1
    elif basisfunc=='quad': M=2
    else: raise Exception('Value "%s" for basisfunc must be either "lin" or "quad"'%(basisfunc))
    # parameters
    print '======Training======'
    print 'lambda = ' + str(lamduh)
    # load data from csv files
    train = loadtxt('data/data_'+name+'_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3]

    # Carry out training.
    phi = makePhi(X,M)
    n,m = phi.shape
    result = logreg.logreg(phi, Y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=optimizePrint)
    w = result[0][:m]
    b = result[0][m]

    # Define the predictLR(x) function, which uses trained parameters
    predictLR = makePredictor(w,b,M,mode='logreg')

    # get training error
    tErr = getError(X, Y, w, b, mode='logreg')

    # plot training results
    if plot:
        plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train with ${\lambda}=%s$' %(lamduh))

    print '======Validation======'
    # load data from csv files
    validate = loadtxt('data/data_'+name+'_validate.csv')
    X = validate[:,0:2]
    Y = validate[:,2:3]

    # plot validation results
    if plot:
        plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate with ${\lambda}=%s$' %(lamduh))

    # get validation error
    vErr = getError(X, Y, w, b, mode='logreg')

    ## return training and validation error
    return numpy.array([tErr, vErr])

def lambdaSweep():
    problemClass = "lr"
    varName = "$\lambda$"

    ## try an exponential sweep up to around 250
    lambdaVals = numpy.array([0] + [10**i for i in numpy.arange(-4,2.5,0.2)])

    ## first try with linear basis functions
    resultsLin = numpy.array([LRTrain({'lamduh':lamduh}, problemClass=problemClass,basisfunc='lin', plot=False, printInfo=False)() for lamduh in lambdaVals])
    trainingErrLin = resultsLin[:,0]
    validationErrLin = resultsLin[:,1]
    plotTVError(lambdaVals, trainingErrLin, validationErrLin, problemClass=problemClass, varName=varName, linQuad='linear', extra='')

    # now try with quadratic basis function
    resultsQuad = numpy.array([LRTrain({'lamduh':lamduh}, basisfunc='quad', plot=False, printInfo=False)() for lamduh in lambdaVals])
    trainingErrQuad = resultsQuad[:,0]
    validationErrQuad = resultsQuad[:,1]
    plotTVError(lambdaVals, trainingErrQuad, validationErrQuad, problemClass=problemClass, varName=varName, linQuad='quadratic', extra='')

    return lambdaVals, trainingErrLin, validationErrLin, trainingErrQuad, validationErrQuad

if __name__ == '__main__':
    name='ls'
    ## a simple training test
    print LRTrain({'lamduh':0.01}, printInfo=True)()

    ## the good stuff
    lambdaSweep()
    pl.show()

