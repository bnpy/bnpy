import argparse
import numpy as np
import bnpy
from matplotlib import pylab

from bnpy.viz import GaussViz

DefaultPlan = dict(
    btargetCompID=0,
    bcreationProposalName='randomSplit',
    btargetMaxSize=200)

def showBirthProposal(
        curModel=None, propModel=None,
        Plan=None,
        origK=0,
        propK=0,
        **kwargs):
    ''' Show before/after images of learned comps for a birth proposal.

    Post Condition
    --------------
    Figure is displayed, but not blocking execution.
    '''
    compsToHighlight = np.arange(origK, propK)
    bnpy.viz.PlotComps.plotCompsFromHModel(
        hmodel=propModel, compsToHighlight=compsToHighlight)
    pylab.show(block=0)


def showBirthBeforeAfter(**kwargs):
    ''' Show before/after images of learned comps for a birth proposal.

    Post Condition
    --------------
    Figure is displayed, but not blocking execution.
    '''
    if str(type(kwargs['curModel'].obsModel)).count('Gauss') > 0:
        _viz_Gauss_before_after(**kwargs)
    else:
        _viz_Mult(**kwargs)
    pylab.show(block=0)


def _viz_Gauss_before_after(
        curModel=None, propModel=None,
        curSS=None, propSS=None,
        Plan=None,
        propLscore=None, curLscore=None,
        Data_b=None, Data_t=None, 
        **kwargs):
    pylab.subplots(
        nrows=1, ncols=2, figsize=(8, 4), num=1)
    h1 = pylab.subplot(1, 2, 1)
    h1.clear()
    GaussViz.plotGauss2DFromHModel(
        curModel, compsToHighlight=Plan['btargetCompID'], figH=h1)
    if curLscore is not None:
        pylab.title('%.4f' % (curLscore))

    h2 = pylab.subplot(1, 2, 2, sharex=h1, sharey=h1)
    h2.clear()
    newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
    GaussViz.plotGauss2DFromHModel(
        propModel, compsToHighlight=newCompIDs, figH=h2, Data=Data_t)
    if propLscore is not None:
        pylab.title('%.4f' % (propLscore))
    
        Lgain = propLscore - curLscore
        if Lgain > 0:
            pylab.xlabel('ACCEPT +%.2f' % (Lgain))
        else:
            pylab.xlabel('REJECT %.2f' % (Lgain))
    pylab.draw()
    pylab.subplots_adjust(hspace=0.1, top=0.9, bottom=0.15,
                          left=0.15, right=0.95)

def showBirthFromSavedPath():
    '''
    '''
    pass


def showBirthFromScratch(hmodel=None, Data=None, **Plan):
    '''
    '''
    from BMain import runBirthMove

    LP = hmodel.calc_local_params(Data)
    ResultDict = runBirthMove(
        Data, hmodel, None, LP, **Plan)
    return ResultDict

if __name__ == '__main__':
    hmodel, Info = bnpy.run() # will auto-scrape stdin for all args
    Data = Info['Data']

    Plan = dict(**DefaultPlan)
    Plan.update(Info['UnkArgs'])
    algName = Info['ReqArgs']['algName']
    Plan['PRNG'] = np.random.RandomState(Info['KwArgs'][algName]['algseed'])
    Lscore_gain = list()
    size_grid =  np.asarray(
        [int(x) for x in str(Plan['btargetMaxSize']).split(',')],
        dtype=np.float64)
    for btargetMaxSize in size_grid:
        Plan['btargetMaxSize'] = btargetMaxSize
        Result = showBirthFromScratch(hmodel, Data=Data, **Plan)
        Lscore_gain.append(Result['Lscore_gain'])

    size_grid = size_grid / Data.get_size()
    print(size_grid)
    print(Lscore_gain)

    if size_grid.size > 1:
        xs = np.linspace(0,1,10)
        pylab.plot(xs, np.zeros(xs.size), 'r--')
        pylab.plot(size_grid, Lscore_gain, 'k.-')
        pylab.xlim([0, 1])
        pylab.xlabel('Frac. of total dataset used in birth proposal')
        pylab.ylabel('ELBO gain after truelabels proposal')
        pylab.show()
