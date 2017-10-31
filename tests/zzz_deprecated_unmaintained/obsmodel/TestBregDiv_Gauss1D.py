import numpy as np
from bnpy.viz.PlotUtil import pylab

def calcBregDiv_Gauss1D(x, mu1, mu2, doCorrection=1):
    ''' Calculate breg divergence between data and mean parameter

    Returns
    -------
    div : ND array, same size as x
    '''
    # Parse inputs
    x = np.asarray(x, dtype=np.float64)
    mu1 = np.float64(mu1)
    mu2 = np.float64(mu2)
    sqx = np.square(x, dtype=np.float64)
    sqmu2 = np.square(mu2, dtype=np.float64)
    # Compute divergence
    v = mu1 - sqmu2
    div = 0.5 * np.log(v) + \
        1.0/v * (0.5 * sqx - 0.5 * mu1 + sqmu2 - x * mu2)
    if doCorrection:
        div = 0.5 + 1.0/v * (0.5 * sqx - 0.5 * mu1 + sqmu2 - x * mu2)
        #correction = 0.5 * np.log(v) + \
        #    1.0/v * (0.5 * mu1 - 0.5 * mu1 + sqmu2 - sqmu2) - 0.5
        #div -= correction
    return div

def makePlot(muVals=[(0.01,0), (0.1,0), (1,0), (10,0)], doCorrection=1):
    pylab.figure()
    xgrid = np.linspace(-8, 8, 2000)
    pylab.hold('on')
    pylab.plot(xgrid, np.zeros_like(xgrid), ':', alpha=0.2)
    for mu1, mu2 in muVals:
        ygrid = calcBregDiv_Gauss1D(xgrid, mu1, mu2, doCorrection=doCorrection)
        print(ygrid.min())
        pylab.plot(xgrid, ygrid, label='mu1=% 6.2f mu2=% 6.2f' % (mu1, mu2))
    pylab.legend(loc='lower right')
    pylab.xlim([xgrid.min(), xgrid.max()])
    pylab.ylim([xgrid.min(), xgrid.max()])
    pylab.xlabel('x')
    if doCorrection:
        pylab.ylabel('D(x, \mu) + correction')
    else:
        pylab.ylabel('D(x, \mu)')

if __name__ == "__main__":
    for doC in [1]:
        makePlot(muVals=[(0.01,0), (0.1,0), (1,0), (10,0)], doCorrection=doC)
        pylab.savefig('BregDivGauss1D_fixedMean_doC=%d.eps' % (doC),
            pad_inches=0, bbox_inches='tight')
        makePlot(muVals=[(4+1,-2), (1,0), (4+1,2), (16+1,4)], doCorrection=doC)
        pylab.savefig('BregDivGauss1D_fixedVar_doC=%d.eps' % (doC),
            pad_inches=0, bbox_inches='tight')   
        makePlot(muVals=[(10,0), (4+1,2), (25+0.01,5)], doCorrection=doC)
        pylab.savefig('BregDivGauss1D_general_doC=%d.eps' % (doC),
            pad_inches=0, bbox_inches='tight')
    pylab.show(block=True)
