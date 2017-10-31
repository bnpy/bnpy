import numpy as np
import bnpy
from bnpy.birthmove.SCreateFromScratch import createSplitStats
from bnpy.birthmove.SAssignToExisting import assignSplitStats

def main(nBatch=100, nSetupLaps=1, targetUID=0, **kwargs):

    import AsteriskK8
    Data = AsteriskK8.get_data(nObsTotal=10000)
    Data.alwaysTrackTruth = 1
    DataIterator = Data.to_iterator(nBatch=nBatch, nLap=10)

    hmodel = bnpy.HModel.CreateEntireModel(
        'moVB', 'DPMixtureModel', 'Gauss',
        dict(), dict(ECovMat='eye'), Data)
    hmodel.init_global_params(Data, K=1, initname='kmeans')

    # Do some fixed-truncation local/global steps
    SS = None
    SSmemory = dict()
    for lap in range(nSetupLaps):
        for batchID in range(nBatch):
            Dbatch = DataIterator.getBatch(batchID)

            LPbatch = hmodel.calc_local_params(Dbatch)
            SSbatch = hmodel.get_global_suff_stats(
                Dbatch, LPbatch, doPrecompEntropy=1)

            if batchID in SSmemory:        
                SS -= SSmemory[batchID]
            SSmemory[batchID] = SSbatch
            if SS is None:
                SS = SSbatch.copy()
            else:
                SS += SSbatch

    for batchID in range(nBatch):
        print('batch %d/%d' % (batchID+1, nBatch))
        Dbatch = DataIterator.getBatch(batchID)

        LPbatch = hmodel.calc_local_params(Dbatch)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1)

        if batchID in SSmemory:        
            SS -= SSmemory[batchID]
        SSmemory[batchID] = SSbatch
        if SS is None:
            SS = SSbatch.copy()
        else:
            SS += SSbatch
        
        if batchID == 0:
            xSSbatch = createSplitStats(
                Dbatch, hmodel, LPbatch, curSSwhole=SS,
                creationProposalName='kmeans',
                targetUID=targetUID,
                newUIDs=np.arange(100, 100+15))
            xSS = xSSbatch.copy()
        else:
            xSSbatch = assignSplitStats(
                Dbatch, hmodel, LPbatch, SS, xSS,
                targetUID=targetUID)
            xSS += xSSbatch

        hmodel.update_global_params(SS)

        if batchID < 10 or (batchID + 1) % 10 == 0:
            curLscore = hmodel.calc_evidence(SS=SS)

            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSS)
            propModel = hmodel.copy()
            propModel.update_global_params(propSS)
            propLscore = propModel.calc_evidence(SS=propSS)
    
            print(propSS.N)
            print(' cursize %.1f   propsize %.1f' % (SS.N.sum(), propSS.N.sum()))
            print(' curLscore %.3f' % (curLscore))
            print('propLscore %.3f' % (propLscore))
            if propLscore > curLscore:
                print('ACCEPTED!')
            else:
                print('REJECTED <<<<<<<<<< :(')

if __name__ == '__main__':
    main()
