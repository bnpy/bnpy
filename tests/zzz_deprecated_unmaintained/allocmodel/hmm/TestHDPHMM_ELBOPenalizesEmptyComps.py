import numpy as np
import unittest

import bnpy
import ToyHMMK4
HDPHMM = bnpy.allocmodel.HDPHMM


class TestHDPHMM_ELBOPenalizesEmptyComps(unittest.TestCase):

    def shortDescription(self):
        return None

    def setUp(self):
        Data, trueResp = makeDataAndTrueResp()
        self.Data = Data
        self.trueResp = trueResp

    def test_resp_construction_is_valid(self):
        ''' Verify that all versions of resp matrix have rows that sum-to-one

            This includes the original "true resp" for the dataset,
            and any newly-created resp that adds an empty state or two to trueResp
        '''
        assert np.allclose(1.0, np.sum(self.trueResp, axis=1))

        for kempty in range(0, 4):
            resp = makeNewRespWithEmptyStates(self.trueResp, kempty)
            assert np.allclose(1.0, np.sum(resp, axis=1))
            assert resp.shape[0] == self.trueResp.shape[0]
            assert resp.shape[1] == self.trueResp.shape[1] + kempty

    def test_ELBO_penalizes_empty_comps(self, **kwargs):
        kemptyVals = np.arange(4)
        ELBOVals = np.zeros_like(kemptyVals, dtype=np.float32)
        for ii, kempty in enumerate(kemptyVals):
            resp = makeNewRespWithEmptyStates(self.trueResp, kempty)
            ELBOVals[ii] = resp2ELBO_HDPHMM(self.Data, resp, **kwargs)
        assert np.all(np.diff(ELBOVals) < 0)

    def test_ELBO_penalizes_empty__range_of_hypers(self):
        print('')
        for initprobs in ['uniform', 'bypopularity']:
            print('------------------- initial beta set to %s' % (initprobs))
            print('%5s %5s %5s' % ('alpha', 'gamma', 'kappa'))
            for alpha in [0.1, 0.9, 1.5]:
                for gamma in [1.0, 3.0, 10.0]:
                    for kappa in [1.1, 17.76, 100]:
                        print('%5.2f %5.2f %7.2f' % (alpha, gamma, kappa))

                        self.test_ELBO_penalizes_empty_comps(
                            alpha=alpha, gamma=gamma,
                            hmmKappa=kappa,
                            initprobs=initprobs)


def printProbVector(xvec, fmt='%.4f'):
    xvec = np.asarray(xvec)
    if xvec.ndim == 0:
        xvec = np.asarray([xvec])
    print(' '.join([fmt % (x) for x in xvec]))


def resp2ELBO_HDPHMM(Data, resp, gamma=10, alpha=0.5, hmmKappa=0,
                     initprobs='bypopularity'):
    K = resp.shape[1]
    scaleF = 1

    # Create a new HDPHMM
    amodel = HDPHMM('VB', dict(alpha=alpha, gamma=gamma, hmmKappa=hmmKappa))

    # Set global params so that
    # E[\beta] gives the desired distribution over the topics
    if initprobs == 'bypopularity':
        init_probs = np.sum(resp, axis=0) + gamma
    elif initprobs == 'uniform':
        init_probs = np.ones(K)
    init_probs = init_probs / np.sum(init_probs)
    amodel.set_global_params(beta=init_probs, Data=Data)

    # Create suff stats that summarize the provided resp values.
    # These will remain fixed, since the token assignments are not changing.
    # The suff stat bag is used to update variable 'amodel'
    LP = dict(resp=resp)
    LP = amodel.initLPFromResp(Data, LP, limitMemoryLP=0)
    SS = amodel.get_global_suff_stats(Data, LP, doPrecompEntropy=0)

    # Fill in all values (theta/rho/omega), and calculate the ELBO
    amodel.update_global_params(SS)
    ELBO = amodel.calc_evidence(Data, SS, LP) / scaleF

    # Loop over alternating updates to local and global parameters
    # until we've converged
    prevELBO = -1 * np.inf
    ELBOtrace = list()
    ELBOtrace.append(ELBO)
    while np.abs(ELBO - prevELBO) > 1e-7:
        prevELBO = ELBO

        # Verify that the updates give expected values for "leftover" index
        Ebeta = amodel.get_active_comp_probs()
        betaRem = 1 - np.sum(amodel.get_active_comp_probs())
        betaRemFromInitTheta = amodel.startTheta[-1] / amodel.startAlpha
        betaRemFromTransTheta = amodel.transTheta[0, -1] / alpha
        assert np.allclose(betaRem, betaRemFromInitTheta)
        assert np.allclose(betaRem, betaRemFromTransTheta)

        # Update the global parameters
        # Remember, this call alternates updates to rho/omega and the thetas
        amodel.update_global_params(SS)

        # Calculate the objective, verify it increases monotonically
        ELBO = amodel.calc_evidence(Data, SS, LP) / scaleF
        ELBOtrace.append(ELBO)
        assert (ELBO - prevELBO) > -1e-9  # verify monotonic increase

    return ELBO


def makeDataAndTrueResp(seed=123, seqLens=(100, 90, 80, 70, 60, 50)):
    Data = ToyHMMK4.get_data(seed, seqLens)
    Ztrue = Data.TrueParams['Z']
    K = len(np.unique(Ztrue))  # Num states
    T_all = Ztrue.size
    trueResp = np.ones((T_all, K))
    for t in range(T_all):
        trueResp[t, Ztrue[t]] = 100 - K + 1
    trueResp /= trueResp.sum(axis=1)[:, np.newaxis]
    assert np.allclose(1.0, trueResp.sum(axis=1))
    return Data, trueResp


def makeNewRespWithEmptyStates(resp, nEmpty=1):
    K = resp.shape[1]
    newResp = np.zeros((resp.shape[0],
                        resp.shape[1] + nEmpty))
    newResp[:, :K] = resp
    np.maximum(newResp, 1e-40, out=newResp)
    return newResp


def makeFigure(hmmKappa=0):
    Data, trueResp = makeDataAndTrueResp()

    kemptyVals = np.asarray([0, 1, 2, 3.])
    ELBOVals = np.zeros_like(kemptyVals, dtype=np.float)

    # Iterate over the number of empty states (0, 1, 2, ...)
    for ii, kempty in enumerate(kemptyVals):
        resp = makeNewRespWithEmptyStates(trueResp, kempty)
        ELBOVals[ii] = resp2ELBO_HDPHMM(Data, resp, hmmKappa=hmmKappa)

    # Make largest value the one with kempty=0, to make plot look good
    ELBOVals -= ELBOVals[0]

    # Plot the results
    from matplotlib import pylab
    figH = pylab.figure(figsize=(6, 4))
    plotargs = dict(markersize=10, linewidth=3)
    pylab.plot(kemptyVals, ELBOVals, 'o--', label='HDP surrogate',
               color='b', markeredgecolor='b',
               **plotargs)
    pylab.xlabel('num. empty topics', fontsize=20)
    pylab.ylabel('change in ELBO', fontsize=20)
    B = 0.25
    pylab.xlim([-B, kemptyVals[-1] + B])
    pylab.xticks(kemptyVals)

    axH = pylab.gca()
    axH.tick_params(axis='both', which='major', labelsize=15)
    legH = pylab.legend(loc='upper left', prop={'size': 15})

    figH.subplots_adjust(bottom=0.16, left=0.2)
    pylab.show(block=True)

if __name__ == "__main__":
    makeFigure(hmmKappa=0)
