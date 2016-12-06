import numpy as np
from bnpy.util.NumericUtil import calcRlogR


def evaluateStatesInSingleSeq(Data_n, LP_n,
                              **kwargs):
    '''

    Returns
    -------
    scores : 1D array, size K
        scores[k] gives the score of comp k
        if comp k doesn't appear in the sequence, scores[k] is 0
    '''
    return calcEvalMetricForStates_KL(LP_n=LP_n, **kwargs)


def calcEvalMetricForStates_KL(LP_n=None, **kwargs):
    ''' Compute KL between empirical and model distr. for each state.

    Returns
    -------
    scores : 1D array, size K
        scores[k] gives the score of comp k
        if comp k doesn't appear in the sequence, scores[k] is 0
    '''
    np.maximum(LP_n['resp'], 1e-100, out=LP_n['resp'])
    N = np.sum(LP_n['resp'], axis=0)

    # Pemp_log_Pemp : 1D array, size K
    # equals the (negative) entropy of the empirical distribution
    # = 1/N_k r_nk * log(r_nk) - log(N_k)
    # = - log(N_k) + 1/N_k r_nk log _rnk
    Pemp_log_Pemp = -1 * np.log(N) + 1.0 / N * calcRlogR(LP_n['resp'])

    Pemp_log_Pmodel = 1.0 / N * np.sum(LP_n['resp'] * LP_n['E_log_soft_ev'])

    KLscore = -1 * Pemp_log_Pmodel + Pemp_log_Pemp
    return KLscore
