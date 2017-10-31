import numpy as np
import argparse
import bnpy

from bnpy.util.RandUtil import rotateCovMat
from bnpy.viz.PlotUtil import pylab
from bnpy.ioutil.BNPYArgParser import arglist_to_kwargs
import sys

def main(makeDataset=None, makeModel=None, 
         doViz=True, **kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default='5')
    args, unkList = parser.parse_known_args()
    kwargs = arglist_to_kwargs(unkList, doConvertFromStr=False)
    kwargs.update(args.__dict__)
    tmpmodel, Data, compsToHighlight = testBSelectMethod_Ldata(
        makeDataset=makeDataset,
        makeModel=makeModel, **kwargs)
    if doViz:
        plotComps(tmpmodel, Data, compsToHighlight=compsToHighlight)
    

def makeInitModelWithMergedComps(model, Data, compsToMerge=[(0,1)], **kwargs):
    initZ = Data.TrueParams['Z'].copy()
    if isinstance(compsToMerge, str):
        compsToMerge_new = list()
        for kgroup in compsToMerge.split('/'):
            compsToMerge_new.append(tuple([int(k) for k in kgroup.split(',')]))
        compsToMerge = compsToMerge_new
        print(compsToMerge)
    for knew, ktuple in enumerate(compsToMerge):
        for k in ktuple:
            initZ[initZ == k] = 1000 + knew
    # relabel initZ by unique entries
    uZ = np.unique(initZ)
    initresp = np.zeros((initZ.size, uZ.size))
    for uid, k in enumerate(uZ):
        initresp[initZ == k, uid] = 1.0
    compsToHighlight = np.flatnonzero(uZ >= 1000)

    initLP = dict(resp=initresp)
    initSS = model.get_global_suff_stats(Data, initLP)
    tmpmodel = model.copy()
    tmpmodel.update_global_params(initSS)
    for aiter in range(10):
        LP = tmpmodel.calc_local_params(Data)
        SS = tmpmodel.get_global_suff_stats(Data, LP)
        tmpmodel.update_global_params(SS)
    return tmpmodel, SS, compsToHighlight


def testBSelectMethod_Ldata(
        makeDataset=None,
        makeModel=None,
        **kwargs):

    Data = makeDataset(**kwargs)
    model = makeModel(Data, **kwargs)

    tmpmodel, SS, compsToHighlight_true = makeInitModelWithMergedComps(
        model, Data, **kwargs)
    Lvec = tmpmodel.obsModel.calcELBO_Memoized(SS=SS, returnVec=1)

    pvec = -1 * Lvec.copy()
    pvec /= SS.N
    pvec -= pvec.max()
    pvec = np.exp(SS.N.sum() * pvec)
    pvec /= pvec.sum()

    print('  N: ' + ' '.join(['% 9.0f' % (Nk) for Nk in SS.N]))

    print('  L: ' + ' '.join(['% .2e' % (Lk) for Lk in Lvec]))
    LvecN = -1* Lvec / SS.N
    print('L/N: ' + ' '.join(['% .2e' % (Lk) for Lk in LvecN]))

    print('  p: ' + ' '.join(['% 9.2f' % (pk) for pk in pvec]))

    compsToHighlight = np.argmax(pvec)
    print(compsToHighlight)
    print(compsToHighlight_true)
    return tmpmodel, Data, compsToHighlight

def plotComps(tmpmodel, Data, compsToHighlight=None):
    bnpy.viz.PlotComps.plotCompsFromHModel(tmpmodel, Data=Data,
        compsToHighlight=compsToHighlight)
    pylab.show()


