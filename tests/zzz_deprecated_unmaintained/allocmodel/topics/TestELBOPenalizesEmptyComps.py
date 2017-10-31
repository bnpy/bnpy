import numpy as np
import unittest
import sys
import os

import bnpy
import BarsK10V900
HDPTopicModel = bnpy.allocmodel.HDPTopicModel

filepath = os.path.sep.join(__file__.split(os.path.sep)[:-1])
sys.path.append(os.path.join(filepath, 'HDP-point-estimation'))
import HDPPE
HDPPE = HDPPE.HDPPE


class TesELBOPenalizesEmptyComps(unittest.TestCase):

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
            ELBOVals[ii] = resp2ELBO_HDPTopicModel(self.Data, resp, **kwargs)
        assert np.all(np.diff(ELBOVals) < 0)

    def test_ELBO_penalizes_empty__range_of_hypers(self):
        print('')
        print('%5s %5s' % ('alpha', 'gamma'))
        for alpha in [0.1, 0.5, 0.9, 1.5]:
            for gamma in [1.0, 3.0, 10.0]:
                print('%5.2f %5.2f' % (alpha, gamma))

                self.test_ELBO_penalizes_empty_comps(alpha=alpha, gamma=gamma)


def printProbVector(xvec, fmt='%.4f'):
    xvec = np.asarray(xvec)
    if xvec.ndim == 0:
        xvec = np.asarray([xvec])
    print(' '.join([fmt % (x) for x in xvec]))


def resp2ELBO_HDPTopicModel(Data, resp,
                            gamma=10, alpha=0.5, initprobs='fromdata',
                            doPointEstimate=False,
                            **kwargs):
    Ktrue = 10
    K = resp.shape[1]
    scaleF = Data.word_count.sum()

    # Create a new HDPHMM
    # with initial global params set so we have a uniform distr over topics
    if doPointEstimate:
        amodel = HDPPE('VB', dict(alpha=alpha, gamma=gamma))
    else:
        amodel = HDPTopicModel('VB', dict(alpha=alpha, gamma=gamma))

    if initprobs == 'fromdata':
        init_probs = np.sum(resp, axis=0) + gamma
    elif initprobs == 'uniform':
        init_probs = np.ones(K)
    init_probs = init_probs / np.sum(init_probs)
    amodel.set_global_params(K=K, beta=init_probs, Data=Data)
    estBeta = amodel.get_active_comp_probs()

    print('doPointEstimate ', doPointEstimate)
    print('first 3 active: ', estBeta[:3])
    print('last  3 active: ', estBeta[Ktrue - 2:Ktrue])
    print('          junk: ', estBeta[Ktrue:])
    # Create a local params dict and suff stats
    # These will remain fixed, used to update amodle
    LP = dict(resp=resp)
    LP = amodel.initLPFromResp(Data, LP)
    SS = amodel.get_global_suff_stats(Data, LP, doPrecompEntropy=0)

    # Fill in all values (rho/omega)
    amodel.update_global_params(SS)

    # Loop over alternating updates to local and global parameters
    # until we've converged
    prevELBO = -1 * np.inf
    ELBO = 0
    while np.abs(ELBO - prevELBO) > 1e-7:
        prevELBO = ELBO

        LP = amodel.updateLPGivenDocTopicCount(LP, LP['DocTopicCount'])
        SS = amodel.get_global_suff_stats(Data, LP, doPrecompEntropy=0)

        Ebeta = amodel.get_active_comp_probs()
        betaRem = 1 - np.sum(amodel.get_active_comp_probs())
        if doPointEstimate:
            betaRemFromTheta = LP['eta0'][0, -1] / alpha
        else:
            betaRemFromTheta = LP['thetaRem'] / alpha
        assert np.allclose(betaRem, betaRemFromTheta)

        amodel.update_global_params(SS)
        ELBO = amodel.calc_evidence(Data, SS, LP) / scaleF

    print(amodel.gamma, amodel.alpha, initprobs, Data.nDoc)
    return ELBO


def makeDataAndTrueResp(seed=123, nDocTotal=10, **kwargs):
    Data = BarsK10V900.get_data(seed, nDocTotal=nDocTotal, nWordsPerDoc=200)
    trueResp = Data.TrueParams['resp'].copy()
    return Data, trueResp


def makeNewRespWithEmptyStates(resp, nEmpty=1):
    K = resp.shape[1]
    newResp = np.zeros((resp.shape[0],
                        resp.shape[1] + nEmpty))
    newResp[:, :K] = resp
    np.maximum(newResp, 1e-40, out=newResp)
    return newResp


def makeFigure(**kwargs):
    Data, trueResp = makeDataAndTrueResp(**kwargs)

    kemptyVals = np.asarray([0, 1, 2, 3.])
    ELBOVals = np.zeros_like(kemptyVals, dtype=np.float)
    PointEstELBOVals = np.zeros_like(kemptyVals, dtype=np.float)

    # Iterate over the number of empty states (0, 1, 2, ...)
    for ii, kempty in enumerate(kemptyVals):
        resp = makeNewRespWithEmptyStates(trueResp, kempty)
        PointEstELBOVals[ii] = resp2ELBO_HDPTopicModel(
            Data,
            resp,
            doPointEstimate=1,
            **kwargs)
        ELBOVals[ii] = resp2ELBO_HDPTopicModel(Data, resp, **kwargs)

    # Make largest value the one with kempty=0, to make plot look good
    PointEstELBOVals -= PointEstELBOVals[0]
    ELBOVals -= ELBOVals[0]

    # Rescale so that yaxis has units on order of 1, not 0.001
    scale = np.max(np.abs(ELBOVals))
    ELBOVals /= scale
    PointEstELBOVals /= scale

    # Set buffer-space for defining plotable area
    xB = 0.25
    B = 0.19  # big buffer for sides where we will put text labels
    b = 0.01  # small buffer for other sides
    TICKSIZE = 30
    FONTSIZE = 40
    LEGENDSIZE = 30
    LINEWIDTH = 4

    # Plot the results
    figH = pylab.figure(figsize=(9.1, 6))
    axH = pylab.subplot(111)
    axH.set_position([xB, B, (1 - xB - b), (1 - B - b)])

    plotargs = dict(markersize=20, linewidth=LINEWIDTH)
    pylab.plot(kemptyVals, PointEstELBOVals, 'v-', label='HDP point est',
               color='b', markeredgecolor='b',
               **plotargs)
    pylab.plot(kemptyVals, np.zeros_like(kemptyVals), 's:', label='HDP exact',
               color='g', markeredgecolor='g',
               **plotargs)
    pylab.plot(kemptyVals, ELBOVals, 'o--', label='HDP surrogate',
               color='r', markeredgecolor='r',
               **plotargs)

    pylab.xlabel('num. empty topics', fontsize=FONTSIZE)
    pylab.ylabel('change in ELBO', fontsize=FONTSIZE)
    xB = 0.25
    pylab.xlim([-xB, kemptyVals[-1] + xB])
    pylab.xticks(kemptyVals)
    pylab.yticks([-1, 0, 1])

    axH = pylab.gca()
    axH.tick_params(axis='both', which='major', labelsize=TICKSIZE)
    legH = pylab.legend(loc='upper left', prop={'size': LEGENDSIZE})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--initprobs', type=str, default='fromdata')
    parser.add_argument('--nDocTotal', type=int, default=10)
    args = parser.parse_args()

    from matplotlib import pylab
    pylab.rcParams['ps.useafm'] = True
    makeFigure(**args.__dict__)

    pylab.show(block=False)
    keypress = input('Press y to save, any other key to close >>')
    if keypress.count('y'):
        pylab.savefig(
            'changeInELBOVsNumEmpty.eps',
            #bbox_inches='tight',
            format='eps')
