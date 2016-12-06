import numpy as np
from bnpy.viz.PlotUtil import pylab

import bnpy
import StarCovarK5
import bnpy.init.FromScratchGauss as FSG

def makeModelForData(Data, gamma=10, ECovMat='eye', sF=0.5, nu=0):
    aDict = dict(gamma=gamma)
    oDict = dict(ECovMat=ECovMat, sF=sF, nu=nu)
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'DPMixtureModel', 'ZeroMeanGauss',
        aDict, oDict, Data)
    return hmodel

def calcBregDiv_ZeroMean(x, mu1, B=1e-10, nu=2, justMahalTerm=0):
    ''' Calculate breg divergence between data and mean parameter

    Returns
    -------
    div : ND array, same size as x
    '''
    xsq = np.square(x, dtype=np.float64)
    xsq = (xsq + B) / (nu - 1)
    mahalDist = 0.5 * (xsq)/mu1
    if justMahalTerm:
        div = mahalDist
    else:
        div = -0.5 - 0.5 * np.log(xsq) + 0.5 * np.log(mu1) + mahalDist
    return div

def makePlot(muVals=[0.01, 0.1, 1, 10], B=1e-10, nu=2, justMahalTerm=0):
    pylab.figure()
    xgrid = np.linspace(0, 8, 2000)
    pylab.hold('on')
    for mu in muVals:
        ygrid = calcBregDiv_ZeroMean(
            xgrid, mu, 
            B=B, nu=nu,
            justMahalTerm=justMahalTerm)
        pylab.plot(xgrid, ygrid,
            linewidth=2,
            label='\mu=%6.2f' % (mu))
    pylab.legend(loc='upper right')
    pylab.xlim([-0.1, xgrid.max()])
    pylab.ylim([-0.1, xgrid.max()])
    pylab.xlabel('x')
    pylab.ylabel('D(x, \mu)')
    pylab.title('B=%s  nu=%s' % (str(B), str(nu)))


if __name__ == "__main__":
    '''
    Data = StarCovarK5.get_data()
    hmodel = makeModelForData(Data)
    Mu = FSG.calcClusterMean_ZeroMeanGauss(Data.X, hmodel)
    Div = FSG.calcBregDiv_ZeroMeanGauss(Data.X, Mu)
    initSS, Info = FSG.initSSByBregDiv_ZeroMeanGauss(
        Dslice=Data, curModel=hmodel, K=7,
        b_initHardCluster=1)
    '''    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nu', type=str, default='3')
    parser.add_argument('--B', type=str, default='1e-10')
    args = parser.parse_args()
    for B in args.B.split(','):
        for nu in args.nu.split(','):
            B = float(B)
            nu = float(nu)
            makePlot(justMahalTerm=0, nu=nu, B=B)
            savefilename = 'BregDiv_B=%s_nu=%s.eps' % (
                str(B), str(nu))
            pylab.savefig(savefilename, bbox_inches='tight', pad_inches=0)
    pylab.show(block=True)
