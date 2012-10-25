from numpy import *
from plotBoundary import *
# import your SVM training code
import svm
import numpy
from utils import *
from svm import *
from plotBoundary import *

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
            self.result = svm.dual(phi,Y,C,kernel)
            self.alphaD = numpy.array(self.result['x'])
            w,b,self.dualS,self.dualM = svm.dualWeights(phi, Y, kernel, self.alphaD, C)
        self.slack = numpy.array(self.result['z'])[:-n]
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

    

def cSweep():
    problemClass = "svm"
    varName = "C"

    ## try an exponential sweep up to around 250
    cVals = numpy.array([0] + [10**i for i in numpy.arange(-4,2.5,0.2)])

    # ## first try the primal with linear basis functions, linear kernel
    # resultsLin = numpy.array([SVMTrain({'primal':True, 'C':C}, basisfunc='lin', plot=False, printInfo=False)() for c in cVals])
    # trainingErrLin = resultsLin[:,0]
    # validationErrLin = resultsLin[:,1]
    # plotTVError(lambdaVals, trainingErrLin, validationErrorLin, problemClass=problemClass, varName=varName, linQuad='linear', extra='')

    # # now try the primal with quadratic basis function, linear kernel
    # resultsQuad = numpy.array([SVMTrain({'primal':True, 'C':C}, basisfunc='quad', plot=False, printInfo=False)() for c in cVals])
    # trainingErrQuad = resultsQuad[:,0]
    # validationErrQuad = resultsQuad[:,1]
    # plotTVError(lambdaVals, trainingErrQuad, validationErrorQuad, problemClass=problemClass, varName=varName, linQuad='quadratic', extra='')

    def plotTVErrorWrapper(cVals, primal, kernelName, kernel, linQuad, problemClass, varName):
        print "primal=%s, linQuad=%s, kernelName=%s" %(primal, linQuad, kernelName)
        results = numpy.array([SVMTrain({'primal':(True if primal=='primal' else False), varName:C, 'kernel':kernel}, basisfunc=linQuad, plot=False, printInfo=True)() for C in cVals])
        trainingErr = results[:,0]
        validationErr = results[:,1]
        plotTVError(cVals, trainingErr, validationErr, problemClass=problemClass, varName=varName, linQuad=('quadratic' if linQuad=='quad' else 'linear'), extra=' and a %s kernel, %s form' %(kernelName, primal))
        return numpy.array([results, trainingErr, validationErr])

    kernels = [('linear',linearKernel), ('quadratic',squaredKernel), ('Gaussian',gaussianKernel)]
    lq = ['lin','quad']
    p = ['primal','dual']
    ## plot and return all 3*2*2 = 12 possibilities
    result = numpy.array([plotTVErrorWrapper(cVals, primal, kernelName, kernel, linQuad, problemClass, varName) for kernelName, kernel in kernels for linQuad in lq for primal in p])

    return cVals, result


if __name__=='__main__':
    #print SVMTrain({'primal':False,'C':0.1, 'kernel':gaussianKernel}, problemClass='svm', basisfunc='quad', printInfo=True, plot=True)()
    #gaussianKernel = Kernel(lambda a,b: exp(-0.3*((a-b).T.dot(a-b))))
    #a=SVMTrain({'primal':False,'C':10, 'kernel':squaredKernel}, problemClass='svm', basisfunc='lin', printInfo=True, plot=True)
    #e = a()
    #print e
    #cSweep()

    ## a simple check
    dummyX = numpy.array([[1,1],
                          [2,2]])
    dummyY = numpy.array([[-1],
                          [1]])

    dummyX = numpy.array([[1,1],[0.9,1.02],[1.02,0.93],[1,1],[0.9,1.02],[1.02,0.93],[1,1],[0.9,1.02],[1.02,0.93],
                          [2.02,2.2],[2.3,2.4],[2,2],[2.02,2.2],[2.3,2.4],[2,2],[2.02,2.2],[2.3,2.4],[2,2],
                          [1.4, 1.4],
                          [1.6,1.6]
                          ])
    dummyY = numpy.array([[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],[-1],
                          [1],[1],[1],[1],[1],[1],[1],[-1],[1],
                          [-1],
                          [1]])

    b=SVMTrain({'primal':False,'C':1, 'kernel':linearKernel}, problemClass='svm', basisfunc='lin', printInfo=False, plot=True)
    print b._computeError(dummyX, dummyY)
    #b=SVMTrain({'primal':False,'C':1000, 'kernel':linearKernel}, problemClass='svm', basisfunc='lin', printInfo=False, plot=True)
    #print b()
