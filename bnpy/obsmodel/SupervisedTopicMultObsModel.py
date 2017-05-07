import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from AbstractObsModel import AbstractObsModel
import copy

import GaussObsModel as GaussX
from MultObsModel import MultObsModel
import SMLogisticRegressYFromFixedTopicModelDiag as RegressY

class SupervisedTopicMultObsModel(MultObsModel):

    ''' Model for producing 1D observations from modeled covariates

    Attributes for DiagGauss Prior
    ------------------------------
    nu : scalar positive float
        degrees of freedom for precision-matrix random variable L
    B : 2D array, size D x D
        determines mean of the precision-matrix random variable L
    m : 1D array, size D
        mean of the location parameter mu
    kappa : scalar positive float
        additional precision for location parameter mu

    Attributes for Regression Prior
    -------------------------------
    w_E : 1D array, size E
        mean of the regression weights
    P_EE : 2D array, size E x E
        precision matrix for regression weights
    nu : positive float
        effective sample size of prior on regression precision
    tau : positive float
        effective scale parameter of prior on regression precision

    Attributes for Point Estimation
    -------------------------------
    TODO

    Attributes for Approximate Posterior
    ------------------------------------
    Same names as for the Prior
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
            print 'Current acc:', np.sum((np.round(Ypred) == Data.Y).astype(float)) / Ypred.shape[0]
            print 'Current mse:', np.sum((Ypred - Data.Y) ** 2) / Ypred.shape[0]
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

    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
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
        elbo = super(SupervisedTopicMultObsModel, self).calcELBO_Memoized(SS, returnVec, afterMStep, **kwargs)
        elbo += RegressY.calcELBOFromSSAndPost(SS, Post=self.Post, Prior=self.Prior, returnVec=returnVec, afterMStep=afterMStep, **kwargs)
        return elbo

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        assert False

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        assert False

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def predictYFromLP(Data, LP, Post, **kwargs):
    nDoc = Data.nDoc
    Y = np.zeros(nDoc)
    for d in xrange(nDoc):
    	x = LP['DocTopicCount'][d,:]
    	x = x / np.sum(x)
    	Y[d] = sigmoid(np.dot(x, Post.w_m))
    return Y





