from builtins import *
import numpy as np
from . import HMMUtil
from bnpy.util import digamma, gammaln
from bnpy.util.StickBreakUtil import rho2beta
from bnpy.allocmodel.topics.HDPTopicUtil import c_Beta, c_Dir
from bnpy.allocmodel.topics import OptimizerRhoOmega

ELBOTermDimMap = dict(
    Htable=('K', 'K'),
    Hstart=('K'),
)


def calcELBO(**kwargs):
    """ Calculate ELBO objective for provided model state.

    Returns
    -------
    L : scalar float
        L is the value of the objective function at provided state.
    """
    Llinear = calcELBO_LinearTerms(**kwargs)
    Lnon = calcELBO_NonlinearTerms(**kwargs)
    if 'todict' in kwargs and kwargs['todict']:
        Llinear.update(Lnon)
        return Llinear
    return Lnon + Llinear


def calcELBO_LinearTerms(SS=None,
                         StartStateCount=None, TransStateCount=None,
                         rho=None, omega=None,
                         Ebeta=None,
                         startTheta=None, transTheta=None,
                         startAlpha=0, alpha=0, kappa=None, gamma=None,
                         afterGlobalStep=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms that are linear in suff stats.

    Returns
    -------
    L : scalar float
        L is sum of any term in ELBO that is const/linear wrt suff stats.
    """
    Ltop = L_top(rho=rho, omega=omega, alpha=alpha, gamma=gamma,
                 kappa=kappa, startAlpha=startAlpha)
    LdiffcDir = - c_Dir(transTheta) - c_Dir(startTheta)
    if afterGlobalStep:
        if todict:
            return dict(
                Lalloc=Ltop+LdiffcDir,
                Lslack=0)
        return Ltop + LdiffcDir

    K = rho.size
    if Ebeta is None:
        Ebeta = rho2beta(rho, returnSize='K+1')

    if SS is not None:
        StartStateCount = SS.StartStateCount
        TransStateCount = SS.TransStateCount
    # Augment suff stats to be sure have 0 in final column,
    # which represents inactive states.
    if StartStateCount.size == K:
        StartStateCount = np.hstack([StartStateCount, 0])
    if TransStateCount.shape[-1] == K:
        TransStateCount = np.hstack([TransStateCount, np.zeros((K, 1))])

    LstartSlack = np.inner(
        StartStateCount + startAlpha * Ebeta - startTheta,
        digamma(startTheta) - digamma(startTheta.sum())
        )

    alphaEbetaPlusKappa = alpha * np.tile(Ebeta, (K, 1))
    alphaEbetaPlusKappa[:, :K] += kappa * np.eye(K)
    digammaSum = digamma(np.sum(transTheta, axis=1))
    LtransSlack = np.sum(
        (TransStateCount + alphaEbetaPlusKappa - transTheta) *
        (digamma(transTheta) - digammaSum[:, np.newaxis])
        )

    if todict:
        return dict(
            Lalloc=Ltop+LdiffcDir,
            Lslack=LstartSlack+LtransSlack)
    return Ltop + LdiffcDir + LstartSlack + LtransSlack


def calcELBO_NonlinearTerms(Data=None, SS=None, LP=None,
                            resp=None, respPair=None,
                            Htable=None, Hstart=None,
                            returnMemoizedDict=0, todict=0, **kwargs):
    """ Calculate ELBO objective terms non-linear in suff stats.
    """
    if Htable is None:
        if SS is not None and SS.hasELBOTerm('Htable'):
            Htable = SS.getELBOTerm('Htable')
        elif LP is not None:
            if 'respPair' in LP:
                Htable = HMMUtil.calc_Htable(LP['respPair'])
            else:
                Htable = np.sum(LP['Htable'], axis=0)
        else:
            Htable = HMMUtil.calc_Htable(respPair)

    if Hstart is None:
        if SS is not None and SS.hasELBOTerm('Hstart'):
            Hstart = SS.getELBOTerm('Hstart')
        else:
            if LP is not None:
                resp = LP['resp']
            Hstart = HMMUtil.calc_Hstart(resp, Data=Data)

    if returnMemoizedDict:
        return dict(Hstart=Hstart, Htable=Htable)
    Lentropy = Htable.sum() + Hstart.sum()
    # For stochastic (soVB), we need to scale up the entropy
    # Only used when --doMemoELBO is set to 0 (not recommended)
    if SS is not None and SS.hasAmpFactor():
        Lentropy *= SS.ampF

    if todict:
        return dict(Lentropy=Lentropy)
    return Lentropy


def L_top(rho=None, omega=None, alpha=None,
          gamma=None, kappa=0, startAlpha=0, **kwargs):
    ''' Evaluate the top-level term of the surrogate objective
    '''
    if startAlpha == 0:
        startAlpha = alpha

    K = rho.size
    eta1 = rho * omega
    eta0 = (1 - rho) * omega
    digamma_omega = digamma(omega)
    ElogU = digamma(eta1) - digamma_omega
    Elog1mU = digamma(eta0) - digamma_omega
    diff_cBeta = K * c_Beta(1.0, gamma) - c_Beta(eta1, eta0)

    tAlpha = K * K * np.log(alpha) + K * np.log(startAlpha)
    if kappa > 0:
        coefU = K + 1.0 - eta1
        coef1mU = K * OptimizerRhoOmega.kvec(K) + 1.0 + gamma - eta0
        sumEBeta = np.sum(rho2beta(rho, returnSize='K'))
        tBeta = sumEBeta * (np.log(alpha + kappa) - np.log(kappa))
        tKappa = K * (np.log(kappa) - np.log(alpha + kappa))
    else:
        coefU = (K + 1) + 1.0 - eta1
        coef1mU = (K + 1) * OptimizerRhoOmega.kvec(K) + gamma - eta0
        tBeta = 0
        tKappa = 0

    diff_logU = np.inner(coefU, ElogU) \
        + np.inner(coef1mU, Elog1mU)
    return tAlpha + tKappa + tBeta + diff_cBeta + diff_logU


def calcELBOForSingleSeq_FromLP(Data_n, LP_n, hmodel,
                                nExtraGlobalSteps=4,
                                **kwargs):
    ''' Compute HDPHMM objective score for single sequence.

    Performs relevant global steps to get model parameters.

    Returns
    -------
    L : scalar score for current sequence
    '''
    assert Data_n.nDoc == 1
    tempModel = hmodel.copy()
    SS_n = tempModel.get_global_suff_stats(Data_n, LP_n,
                                           doPrecompEntropy=1)
    tempModel.update_global_params(SS_n)
    for giter in range(nExtraGlobalSteps):
        tempModel.allocModel.update_global_params(SS_n)

    L_n = tempModel.calc_evidence(SS=SS_n)
    return L_n
