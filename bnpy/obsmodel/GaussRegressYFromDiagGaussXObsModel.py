from builtins import *
import numpy as np
import scipy.linalg
from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, fillSharedMemArray
from bnpy.util.SparseRespStatsUtil import calcSpRXXT
from .AbstractObsModel import AbstractObsModel

from . import GaussRegressYFromFixedXObsModel as RegressY
from . import DiagGaussObsModel as DiagGaussX

class GaussRegressYFromDiagGaussXObsModel(AbstractObsModel):

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

    def __init__(self, inferType='VB', D=0, Data=None, **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Returns
        -------
        obsmodel : bare observation model
            Resulting object lacks either EstParams or Post attributes.
            which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.dim
        else:
            self.D = int(D)
        self.E = self.D + 1
        self.K = 0
        self.inferType = inferType
        self.Cache = dict()
        PriorArgs['D'] = self.D
        PriorArgs['E'] = self.E
        self.Prior = DiagGaussX.createParamBagForPrior(
            Data, **PriorArgs)
        self.Prior = RegressY.createParamBagForPrior(
            Data, Prior=self.Prior, **PriorArgs)

    def get_name(self):
        return 'GaussRegressYFromDiagGaussX'

    def get_info_string(self):
        return 'Gaussian regression model for 1D y from DiagGauss x'

    def get_info_string_prior(self):
        return DiagGaussX.getStringSummaryOfPrior(self.Prior) \
            + RegressY.getStringSummaryOfPrior(self.Prior)

    def setPostFactors(
            self, obsModel=None, SS=None, LP=None, Data=None,
            **param_kwargs):
        ''' Set attribute Post to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            if hasattr(obsModel, 'Post'):
                self.Post = obsModel.Post.copy()
                self.K = self.Post.K
            else:
                self.setPostFromEstParams(obsModel.EstParams)
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)
        if SS is not None:
            self.updatePost(SS)
        else:
            self.Post = DiagGaussX.packParamBagForPost(**param_kwargs)
            self.Post = RegressY.packParamBagForPost(
                Post=self.Post, **param_kwargs)

        self.K = self.Post.K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        SS = DiagGaussX.calcSummaryStats(Data, SS, LP, **kwargs)
        SS = RegressY.calcSummaryStats(Data, SS, LP, **kwargs)
        return SS

    def getDatasetScale(self, SS):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        return SS.N.sum() * (SS.D + 1)

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Compute expected log soft evidence of each item under each cluster

        Returns
        -------
        E_log_soft_ev_NK : 2D array, size N x K
        '''
        E_log_soft_ev_NK = DiagGaussX.calcLogSoftEvMatrix_FromPost(
            Data,
            Post=self.Post)
        if hasattr(Data, 'Y'):
            # Only incorporate Y attribute if present
            E_log_soft_ev_NK = RegressY.calcLogSoftEvMatrix_FromPost(
                Data,
                Post=self.Post,
                E_log_soft_ev_NK=E_log_soft_ev_NK)
        return E_log_soft_ev_NK

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Optimizes the variational objective for approximating the posterior

        Post Condition
        --------------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or SS.K != self.Post.K:
            self.Post = None
        self.Post = DiagGaussX.calcPostParamsFromSS(
            SS=SS, Prior=self.Prior, Post=self.Post)
        self.Post = RegressY.calcPostParamsFromSS(
            SS=SS, Prior=self.Prior, Post=self.Post)
        self.K = SS.K

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
        elbo_XModel = DiagGaussX.calcELBOFromSSAndPost(
            SS=SS,
            Post=self.Post,
            Prior=self.Prior,
            returnVec=returnVec, afterMStep=afterMStep)
        elbo_YModel = RegressY.calcELBOFromSSAndPost(
            SS=SS,
            Post=self.Post,
            Prior=self.Prior,
            returnVec=returnVec, afterMStep=afterMStep)
        return elbo_XModel + elbo_YModel

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        gapX, _, _ = DiagGaussX.calcHardMergeGapForPair(
            SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB)
        gapY, _, _ = RegressY.calcHardMergeGapForPair(
            SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB)
        return gapX + gapY

    def calcHardMergeGap_SpecificPairs(self, SS, PairList):
        ''' Calc change in ELBO for specific list of candidate hard merge pairs

        Returns
        ---------
        Gaps : 1D array, size L
              Gap[j] : scalar change in ELBO after merge of pair in PairList[j]
        '''
        Gaps = np.zeros(len(PairList))
        XcPrior = None
        XcPost_K = [None for k in range(SS.K)]
        YcPrior = None
        YcPost_K = [None for k in range(SS.K)]
        for ii, (kA, kB) in enumerate(PairList):
            gapX, XcPost_K, XcPrior = DiagGaussX.calcHardMergeGapForPair(
                SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB,
                cPrior=XcPrior, cPost_K=XcPost_K)
            gapY, YcPost_K, YcPrior = DiagGaussX.calcHardMergeGapForPair(
                SS=SS, Post=self.Post, Prior=self.Prior, kA=kA, kB=kB,
                cPrior=YcPrior, cPost_K=YcPost_K)
            Gaps[ii] = gapX + gapY
        return Gaps

    def predictClusterSpecificYFromX(self, Data):
        ''' Predict output given input for each cluster

        Returns
        -------
        Y_NK : 2D array, size N x K
            Y_NK[n, k] is equal to E[ y_n | x_n, z_n=k ]
        '''
        return predictClusterSpecificYFromX(
            Data=Data,
            Post=self.Post)

def predictClusterSpecificYFromX(
        Data=None, X_ND=None, X_NE=None, Post=None, **kwargs):
    ''' Predict output given input for each cluster

    Returns
    -------
    Y_NK : 2D array, size N x K
        Y_NK[n, k] is equal to E[ y_n | x_n, z_n=k ]
    '''
    if X_NE is None:
        if Data is not None:
            if isinstance(Data, np.ndarray):
                X_ND = Data
            else:
                X_ND = Data.X

        assert isinstance(X_ND, np.ndarray)
        assert X_ND.ndim == 2
        N = X_ND.shape[0]
        X_NE = np.hstack([X_ND, np.ones(N)[:,np.newaxis]])

    # Verify input data
    assert isinstance(X_NE, np.ndarray)
    assert X_NE.ndim == 2
    assert X_NE[:,-1].min() == 1.0
    assert X_NE[:,-1].max() == 1.0
    N, E = X_NE.shape

    # Make cluster-specific prediction for each data point
    Y_NK = np.dot(X_NE, Post.w_KE.T)
    return Y_NK
