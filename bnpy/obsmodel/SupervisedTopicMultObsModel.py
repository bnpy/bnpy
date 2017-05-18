import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from bnpy.util import lam, eta_update, calc_Zbar_ZZT_manyDocs
from AbstractObsModel import AbstractObsModel
import copy

import GaussObsModel as GaussX
from MultObsModel import MultObsModel
import SMLogisticRegressYFromFixedTopicModelDiag as RegressY

class SupervisedTopicMultObsModel(MultObsModel):

    '''
    KNOWN ISSUES (TODO):
        - Updates and ELBO are (slightly) wrong! Both implicitly use the
            expectation E[Z Z^T], but compute it at E[Z]E[Z]^T, which is
            wrong, though the difference should be small

        - Predictions are also (slighty) wrong! Currently defaults to
            making predictions by using E[w] and the standard sigmoid,
            when it should (probably) use the posterior predictive (bound)
            p(y | x, D)

        - Posterior predictive is wrong! (See the first bullet), also there
            is some weirdness in the code that should be checked out
    '''


    def __init__(self, inferType='VB', **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Returns
        -------
        obsmodel : bare observation model
            Resulting object lacks either EstParams or Post attributes.
            which must be created separately (see init_global_params).
        '''

        super(SupervisedTopicMultObsModel, self).__init__(inferType=inferType, **PriorArgs)
        Data = PriorArgs['Data'] if 'Data' in PriorArgs else None
        if 'Data' in PriorArgs:
            del PriorArgs['Data']
        self.Prior = RegressY.createParamBagForPrior(
            Data, Prior=self.Prior, **PriorArgs)


    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Return
        --------
        SS : SuffStatBag object, with K components.
        '''
        SS = super(SupervisedTopicMultObsModel, self).calcSummaryStats(Data, SS, LP, **kwargs)
        if hasattr(self, 'Post'):
            SS = RegressY.calcSummaryStats(Data, SS, LP, Prior=self.Prior, Post=self.Post, **kwargs)
        else:
            SS = RegressY.calcSummaryStats(Data, SS, LP, Prior=self.Prior, **kwargs)
    
        if hasattr(self.Post, 'w_m'):
            Ypred = predictYFromLP(Data, LP, self.Post)
            print '\tCurrent acc:', np.sum((np.round(Ypred) == Data.Y).astype(float), axis=0) / Ypred.shape[0]
        return SS

    def calc_local_params(self, Data, LP=None, **kwargs):
        """ Calculate local 'likelihood' params for each data item.

        Returns
        -------
        LP : dict
            local parameters as key/value pairs, with fields
            * 'E_log_soft_ev' : 2D array, N x K
                Entry at row n, col k gives (expected value of)
                likelihood that observation n is produced by component k
        """
        LP = super(SupervisedTopicMultObsModel, self).calc_local_params(Data, LP, **kwargs)
       	if hasattr(self, 'Post'):
            LP.update(RegressY.calcLocalParams(Data, Post=self.Post, Prior=self.Prior))
        else:
            LP.update(RegressY.calcLocalParams(Data, Prior=self.Prior))
        return LP

    def setPostFactors(self, w_m=None, **kwargs):
        ''' Set attribute Post to provided values.
        '''
        super(SupervisedTopicMultObsModel, self).setPostFactors(**kwargs)
        if w_m is not None:
            self.Post.setField('w_m', w_m, dims=('K'))

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Optimizes the variational objective for approximating the posterior

        Post Condition
        --------------
        Attributes K and Post updated in-place.
        '''
        super(SupervisedTopicMultObsModel, self).updatePost(SS)
        self.Post = RegressY.calcPostParamsFromSS(SS=SS, Prior=self.Prior, Post=self.Post)

    def calcELBO_Memoized(self, SS, returnVec=0, afterGlobalStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        elbo_K : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        elbox = super(SupervisedTopicMultObsModel, self).calcELBO_Memoized(SS, returnVec, afterGlobalStep, **kwargs)
        elboy = RegressY.calcELBOFromSSAndPost(SS, Post=self.Post, Prior=self.Prior, returnVec=returnVec, afterMStep=afterGlobalStep, **kwargs)
        return elbox + elboy

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        gapx = super(SupervisedTopicMultObsModel, self).calcHardMergeGap(SS, kA, kB)
        gapy = RegressY.calcHardMergeGapForPair(SS=SS, Prior=self.Prior, Post=self.Post, kA=kA, kB=kB)
        return gapx + gapy

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        for ii, (kA, kB) in enumerate(PairList):
            Gaps[ii] = self.calcHardMergeGap(SS, kA, kB)
        return Gaps

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Gap = super(SupervisedTopicMultObsModel, self).calcHardMergeGap_AllPairs(SS)
        for kA in xrange(SS.K):
            for kB in xrange(kA + 1, SS.K):
                Gap[j, k] += RegressY.calcHardMergeGapForPair(SS=SS, Prior=self.Prior, Post=self.Post, kA=kA, kB=kB)
        return Gap


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predictYFromLP(Data, LP, Post, **kwargs):
    nDoc = Data.nDoc
    L = Post.w_m.shape[0]
    Y = np.zeros((nDoc, L))
    for d in xrange(nDoc):
        x = LP['DocTopicCount'][d,:]
        x = x / np.sum(x)

        for i in xrange(L):
            Y[d, i] = sigmoid(np.dot(x, Post.w_m[i]))

    return Y


def log_g(x):
    return -np.log(1.0 + np.exp(-x))

def predictYFromLP_Bound(Data, LP, Post, **kwargs):
    nDoc = Data.nDoc
    L = Post.w_m.shape[0]
    Y = np.zeros((nDoc, L))

    X_all, XXT_all = calc_Zbar_ZZT_manyDocs(LP['resp'], Data.word_count, Data.doc_range)

    for i in xrange(L):
        Sinv, S, w_m = Post.Sinv[i], Post.S[i], Post.w_m[i]
        if Sinv.size == w_m.size:
            Sinv = np.diag(Sinv.flatten())
            S = np.diag(S.flatten())
        
        Sinv_mu = np.dot(Sinv, w_m)
        mu_Sinv_mu = np.dot(Sinv_mu, w_m)

        
        for d in xrange(nDoc):
            X = X_all[d, :]
            XXT = XXT_all[d, :, :]

            eta_t = eta_update(w_m, S, X, XXT)
            eta_t_2 = eta_t ** 2

            Sinv_t = Sinv + 2 * lam(eta_t) * XXT
            mu_t_0 = np.linalg.solve(Sinv_t, Sinv_mu - 0.5 * X)
            mu_t_1 = np.linalg.solve(Sinv_t, Sinv_mu + 0.5 * X)

            common = log_g(eta_t) - eta_t / 2.0 + lam(eta_t) * eta_t_2 - \
                0.5 * mu_Sinv_mu - 0.5 * np.linalg.slogdet(S)[1] - 0.5 * np.linalg.slogdet(Sinv_t)[1]
            y_adj_0 = 0.5 * np.dot(np.dot(Sinv_t, mu_t_0), mu_t_0)
            y_adj_1 = 0.5 * np.dot(np.dot(Sinv_t, mu_t_1), mu_t_1)

            p_y_0 = common + y_adj_0
            p_y_1 = common + y_adj_1

            #TODO: This is reveresed for some reason, check the math
            Y[d, i] = p_y_0 / (p_y_0 + p_y_1)

    return Y





