import numpy as np
import bnpy
from OptimizerForPi import estimatePi2, pi2str
from OptimizerForPi_Doan import estimatePiForDoc

def initMu_BregmanMixture(Data, K, obsModel, seed=0,
        optim_method='frankwolfe'):
    '''
    '''
    PRNG = np.random.RandomState(int(seed))
    X = Data.getDocTypeCountMatrix()
    V = Data.vocab_size
    # Select first cluster mean as uniform distribution
    Mu0 = obsModel.calcSmoothedMu(np.zeros(V))
    # Initialize list to hold all Mu values
    Mu = [None for k in range(K)]
    Mu[0] = Mu0
    chosenZ = np.zeros(K, dtype=np.int32)
    chosenZ[0] = -1
    # Compute minDiv
    minDiv, DivDataVec = obsModel.calcSmoothedBregDiv(
        X=X, Mu=Mu0,
        returnDivDataVec=True,
        return1D=True,
        smoothFrac=1.0)
    Pi = np.ones((X.shape[0], K))
    scoreVsK = list()
    for k in range(1, K):
        sum_minDiv = np.sum(minDiv)        
        scoreVsK.append(sum_minDiv)
        if sum_minDiv == 0.0:
            # Duplicate rows corner case
            # Some rows of X may be exact copies, 
            # leading to all minDiv being zero if chosen covers all copies
            chosenZ = chosenZ[:k]
            for emptyk in reversed(list(range(k, K))):
                # Remove remaining entries in the Mu list,
                # so its total size is now k, not K
                Mu.pop(emptyk)
            assert len(Mu) == chosenZ.size
            break
        elif sum_minDiv < 0 or not np.isfinite(sum_minDiv):
            raise ValueError("sum_minDiv not valid: %f" % (sum_minDiv))
        pvec = minDiv / np.sum(sum_minDiv)
        chosenZ[k] = PRNG.choice(X.shape[0], p=pvec)
        Mu[k] = obsModel.calcSmoothedMu(X[chosenZ[k]])

        # Compute next value of pi
        Pi, minDiv = estimatePiAndDiv_ManyDocs(Data, Mu, Pi, k+1,
            minDiv=minDiv,
            DivDataVec=DivDataVec,
            optim_method=optim_method)

    scoreVsK.append(np.sum(minDiv))
    #assert np.all(np.diff(scoreVsK) >= -1e-6)
    print(scoreVsK)
    return chosenZ, Mu, minDiv, np.sum(DivDataVec), scoreVsK


def estimatePiAndDiv_ManyDocs(Data, Mu, Pi, k, alpha=0.0,
        optim_method='frankwolfe',
        DivDataVec=None,
        minDiv=None):
    '''
    '''
    if minDiv is None:
        minDiv = np.zeros(Data.nDoc)
    if isinstance(Mu, list):
        topics = np.vstack(Mu[:k])    
    else:
        topics = Mu[:k]
    for d in range(Data.nDoc):
        start_d = Data.doc_range[d]
        stop_d = Data.doc_range[d+1]
        wids_d = Data.word_id[start_d:stop_d]
        wcts_d = Data.word_count[start_d:stop_d]
        # Todo: smart initialization of pi??
        piInit = Pi[d, :k].copy()
        piInit[-1] = 0.1
        piInit[:-1] *= 0.9
        assert np.allclose(piInit.sum(), 1.0)
        if optim_method == 'frankwolfe':
            Pi[d, :k], minDiv[d] = estimatePiForDoc(
                ids_d=wids_d, 
                cts_d=wcts_d,
                topics=topics,
                alpha=alpha,
                returnDist=True)
        else:
            Pi[d, :k], minDiv[d], _ = estimatePi2(
                ids_d=wids_d, 
                cts_d=wcts_d,
                topics=topics,
                alpha=alpha,
                scale=1.0,
                piInit=None)
                #piInit=piInit)
        if d == 0:
            print(pi2str(Pi[d,:k]))
    minDiv -= np.dot(np.log(np.dot(Pi[:, :k], topics)), obsModel.Prior.lam)
    if DivDataVec is not None:
        minDiv += DivDataVec
    assert np.min(minDiv) > -1e-6
    np.maximum(minDiv, 0, out=minDiv)
    return Pi, minDiv

if __name__ == '__main__':
    import CleanBarsK10
    Data = CleanBarsK10.get_data(nDocTotal=100, nWordsPerDoc=500)
    K = 3

    #import nips
    #Data = nips.get_data()

    hmodel, Info = bnpy.run(Data, 'DPMixtureModel', 'Mult', 'memoVB', 
        initname='bregmankmeans+0',
        K=K,
        nLap=0)

    obsModel = hmodel.obsModel.copy()
    bestMu = None
    bestScore = np.inf
    nTrial = 1
    for trial in range(nTrial):
        chosenZ, Mu, minDiv, sumDataTerm, scoreVsK = initMu_BregmanMixture(
            Data, K, obsModel, seed=trial)
        score = np.sum(minDiv)
        print("init %d/%d : sum(minDiv) %8.2f" % (trial, nTrial, np.sum(minDiv)))
        if score < bestScore:
            bestScore = score
            bestMu = Mu
            print("*** New best")
    from IPython import embed; embed()
