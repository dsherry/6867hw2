from numpy import *
from plotBoundary import *
# import your LR training code
import scipy

import logreg
from logreg import make_phi

# Define the predictSVM(x) function, which uses trained parameters
def makePredictor(w,b):
    def predict(x):
        #print x.shape, w.shape
        ## transform into feature space
        phi = make_phi(x)
        #print phi.shape, w.shape
        val = logreg.sigmoid(phi.dot(w) + b)
        return 1 if val > 0.5 else -1
    return predict

# parameters
name = 'ls'
print '======Training======'
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
lamduh=.01
phi = make_phi(X)
n,m = phi.shape
result = logreg.logreg(phi, Y, lamduh, opt=scipy.optimize.fmin_bfgs, printInfo=True)
w = result[0][:m]
b = result[0][m]

# Define the predictLR(x) function, which uses trained parameters
predictLR = makePredictor(w,b)

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
