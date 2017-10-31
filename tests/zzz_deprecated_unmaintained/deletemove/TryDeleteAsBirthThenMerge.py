import numpy as np
import bnpy
import BarsK10V900

from bnpy.birthmove import assignSplitStats

LPkwargs = dict(
    nCoordAscentItersLP=100,
    convThrLP=0.001,
    )

def constructXSS(Data, curModel, curSS, curLP):
    ''' Create candidate expansion stats.
    '''
    targetUID = curSS.uids[-1]

    propRemSS = curSS.copy()
    propRemSS.removeComp(uid=targetUID)
    mUIDPairs = list()
    for uid in propRemSS.uids:
        mUIDPairs.append((uid, uid+1000))
    propRemSS.uids = [u+1000 for u in propRemSS.uids]

    xSS = dict()
    xSS[targetUID] = assignSplitStats(
        Data, curModel, curLP, propRemSS,
                curSSwhole=curSS,
                targetUID=targetUID,
                LPkwargs=LPkwargs,
                keepTargetCompAsEmpty=0,
                mUIDPairs=mUIDPairs)
    return xSS

def createJunkModel(Data):
    aDict = dict(alpha=0.5, gamma=0.5)
    oDict = dict(lam=0.1)
    hmodel = bnpy.HModel.CreateEntireModel(
        'VB', 'HDPTopicModel', 'Mult',
        aDict, oDict, Data)
    hmodel.init_global_params(Data, initname='truelabelsandjunk')

    for loopid in range(3):
        LP = hmodel.calc_local_params(Data, **LPkwargs)
        SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
        hmodel.update_global_params(SS)
    LP = hmodel.calc_local_params(Data, **LPkwargs)
    SS = hmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
    return hmodel, SS, LP

def main():
    Data = BarsK10V900.get_data(nDocTotal=500, nWordsPerDoc=400)
    hmodel, SS, LP = createJunkModel(Data)
    xSS = constructXSS(Data, hmodel, SS, LP)
    propSS = SS.copy()
    propSS.replaceCompWithExpansion(uid=10, xSS=xSS[10])
    for (uidA, uidB) in propSS.mUIDPairs:
        propSS.mergeComps(uidA=uidA, uidB=uidB)

    hmodel.update_global_params(SS)
    curLdict = hmodel.calc_evidence(SS=SS, todict=1)

    propModel = hmodel.copy()
    propModel.update_global_params(propSS)
    propLdict = propModel.calc_evidence(SS=propSS, todict=1)

    for key in ['Lslack', 'Lalloc', 'LcDtheta', 'Lentropy', 'Ldata', 'Ltotal']:
        if key.count('_') > 1:
            continue
        print(key)
        print('   cur %.5f' % (curLdict[key]))
        print('  prop %.5f' % (propLdict[key]))

if __name__ == '__main__':
    main()
