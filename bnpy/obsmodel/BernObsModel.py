from builtins import *
import numpy as np
import itertools

from scipy.special import gammaln, digamma

from bnpy.suffstats import ParamBag, SuffStatBag
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D
from bnpy.util import numpyToSharedMemArray, sharedMemToNumpyArray

from .AbstractObsModel import AbstractObsModel

nx = np.newaxis


class BernObsModel(AbstractObsModel):

    ''' Bernoulli data generation model for binary vectors.

    Attributes for Prior (Beta)
    --------
    lam1 : 1D array, size D
        pseudo-count of positive (binary value=1) observations
    lam0 : 1D array, size D
        pseudo-count of negative (binary value=0) observations

    Attributes for k-th component of EstParams (EM point estimates)
    ---------
    phi[k] : 1D array, size D
        phi[k] is a vector of positive numbers in range [0, 1]
        phi[k,d] is probability that dimension d has binary value 1.

    Attributes for k-th component of Post (VB parameter)
    ---------
    lam1[k] : 1D array, size D
    lam0[k] : 1D array, size D
    '''

    def __init__(self, inferType='EM', D=0,
                 Data=None, CompDims=('K',), **PriorArgs):
        ''' Initialize bare obsmodel with valid prior hyperparameters.

        Resulting object lacks either EstParams or Post,
        which must be created separately (see init_global_params).
        '''
        if Data is not None:
            self.D = Data.dim
        elif D > 0:
            self.D = int(D)
        self.K = 0
        self.inferType = inferType
        self.createPrior(Data, **PriorArgs)
        self.Cache = dict()
        if isinstance(CompDims, tuple):
            self.CompDims = CompDims
        elif isinstance(CompDims, str):
            self.CompDims = tuple(CompDims)
        assert isinstance(self.CompDims, tuple)

    def createPrior(
            self, Data, lam1=1.0, lam0=1.0,
            priorMean=None, priorScale=None,
            eps_phi=1e-8, **kwargs):
        ''' Initialize Prior ParamBag attribute.
        '''
        D = self.D
        self.eps_phi = eps_phi
        self.Prior = ParamBag(K=0, D=D)
        if priorMean is None or priorMean.lower().count('none'):
            lam1 = np.asarray(lam1, dtype=np.float)
            lam0 = np.asarray(lam0, dtype=np.float)
        elif isinstance(priorMean, str) and priorMean.count("data"):
            assert priorScale is not None
            priorScale = float(priorScale)
            if hasattr(Data, 'word_id'):
                X = Data.getDocTypeBinaryMatrix()
                dataMean = np.mean(X, axis=0)
            else:
                dataMean = np.mean(Data.X, axis=0)
            dataMean = np.minimum(dataMean, 0.95) # Make prior more smooth
            dataMean = np.maximum(dataMean, 0.05)
            lam1 = priorScale * dataMean
            lam0 = priorScale * (1-dataMean)
        else:
            assert priorScale is not None
            priorScale = float(priorScale)
            priorMean = np.asarray(priorMean, dtype=np.float64)
            lam1 = priorScale * priorMean
            lam0 = priorScale * (1-priorMean)
        if lam1.ndim == 0:
            lam1 = lam1 * np.ones(D)
        if lam0.ndim == 0:
            lam0 = lam0 * np.ones(D)
        assert lam1.size == D
        assert lam0.size == D
        self.Prior.setField('lam1', lam1, dims=('D',))
        self.Prior.setField('lam0', lam0, dims=('D',))

    def get_name(self):
        return 'Bern'

    def get_info_string(self):
        return 'Bernoulli over %d binary attributes.' % (self.D)

    def get_info_string_prior(self):
        msg = 'Beta over %d attributes.\n' % (self.D)
        if self.D > 2:
            sfx = ' ...'
        else:
            sfx = ''
        msg += 'lam1 = %s%s\n' % (str(self.Prior.lam1[:2]), sfx)
        msg += 'lam0 = %s%s\n' % (str(self.Prior.lam0[:2]), sfx)
        msg = msg.replace('\n', '\n  ')
        return msg

    def setupWithAllocModel(self, allocModel):
        ''' Setup expected dimensions of components.

        Args
        ----
        allocModel : instance of bnpy.allocmodel.AllocModel
        '''
        self.CompDims = allocModel.getCompDims()
        assert isinstance(self.CompDims, tuple)

        allocModelName = str(type(allocModel)).lower()
        if allocModelName.count('hdp') or allocModelName.count('topic'):
            self.DataAtomType = 'word'
        else:
            self.DataAtomType = 'doc'

    def setEstParams(self, obsModel=None, SS=None, LP=None, Data=None,
                     phi=None,
                     **kwargs):
        ''' Set attribute EstParams to provided values.
        '''
        self.ClearCache()
        if obsModel is not None:
            self.EstParams = obsModel.EstParams.copy()
            self.K = self.EstParams.K
            return

        if LP is not None and Data is not None:
            SS = self.calcSummaryStats(Data, None, LP)

        if SS is not None:
            self.updateEstParams(SS)
        else:
            self.EstParams = ParamBag(K=phi.shape[0], D=phi.shape[1])
            self.EstParams.setField('phi', phi, dims=('K', 'D',))
        self.K = self.EstParams.K

    def setEstParamsFromPost(self, Post=None):
        ''' Set attribute EstParams based on values in Post.
        '''
        if Post is None:
            Post = self.Post
        self.EstParams = ParamBag(K=Post.K, D=Post.D)
        phi = Post.lam1 / (Post.lam1 + Post.lam0)
        self.EstParams.setField('phi', phi, dims=('K', 'D',))
        self.K = self.EstParams.K

    def setPostFactors(self, obsModel=None, SS=None, LP=None, Data=None,
                       lam1=None, lam0=None, **kwargs):
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
            lam1 = as2D(lam1)
            lam0 = as2D(lam0)
            D = lam1.shape[-1]
            if self.D != D:
                if not lam1.shape[0] == self.D:
                    raise ValueError("Bad dimension for lam1, lam0")
                lam1 = lam1.T.copy()
                lam0 = lam0.T.copy()

            K = lam1.shape[0]
            self.Post = ParamBag(K=K, D=self.D)
            self.Post.setField('lam1', lam1, dims=self.CompDims + ('D',))
            self.Post.setField('lam0', lam0, dims=self.CompDims + ('D',))
        self.K = self.Post.K

    def setPostFromEstParams(self, EstParams, Data=None, nTotalTokens=1,
                             **kwargs):
        ''' Set attribute Post based on values in EstParams.
        '''
        K = EstParams.K
        D = EstParams.D

        WordCounts = EstParams.phi * nTotalTokens
        lam1 = WordCounts + self.Prior.lam1
        lam0 = (1 - WordCounts) + self.Prior.lam0

        self.Post = ParamBag(K=K, D=D)
        self.Post.setField('lam1', lam1, dims=('K', 'D'))
        self.Post.setField('lam0', lam0, dims=('K', 'D'))
        self.K = K

    def calcSummaryStats(self, Data, SS, LP, **kwargs):
        ''' Calculate summary statistics for given dataset and local parameters

        Returns
        --------
        SS : SuffStatBag object, with K components.
        '''
        return calcSummaryStats(Data, SS, LP,
            DataAtomType=self.DataAtomType, **kwargs)

    def calcSummaryStatsForContigBlock(self, Data, a=0, b=0, **kwargs):
        ''' Calculate summary stats for a contiguous block of the data.

        Returns
        --------
        SS : SuffStatBag object, with 1 component.
        '''
        Xab = Data.X[a:b]  # 2D array, Nab x D
        CountON = np.sum(Xab, axis=0)[np.newaxis, :]
        CountOFF = (b - a) - CountON

        SS = SuffStatBag(K=1, D=Data.dim)
        SS.setField('N', np.asarray([b - a], dtype=np.float64), dims='K')
        SS.setField('Count1', CountON, dims=('K', 'D'))
        SS.setField('Count0', CountOFF, dims=('K', 'D'))
        return SS

    def forceSSInBounds(self, SS):
        ''' Force count vectors to remain positive

        This avoids numerical problems due to incremental add/subtract ops
        which can cause computations like
            x = 10.
            x += 1e-15
            x -= 10
            x -= 1e-15
        to be slightly different than zero instead of exactly zero.

        Post Condition
        -------
        Fields Count1, Count0 guaranteed to be positive.
        '''
        np.maximum(SS.Count1, 0, out=SS.Count1)
        np.maximum(SS.Count0, 0, out=SS.Count0)

    def incrementSS(self, SS, k, Data, docID):
        raise NotImplementedError('TODO')

    def decrementSS(self, SS, k, Data, docID):
        raise NotImplementedError('TODO')

    def calcLogSoftEvMatrix_FromEstParams(self, Data, **kwargs):
        ''' Compute log soft evidence matrix for Dataset under EstParams.

        Returns
        ---------
        L : 2D array, N x K
        '''
        logphiT = np.log(self.EstParams.phi.T)  # D x K matrix
        log1mphiT = np.log(1.0 - self.EstParams.phi.T)  # D x K matrix
        return np.dot(Data.X, logphiT) + np.dot(1 - Data.X, log1mphiT)

    def updateEstParams_MaxLik(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the maximum likelihood objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        self.K = SS.K
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)
        phi = SS.Count1 / (SS.Count1 + SS.Count0)
        # prevent entries from reaching exactly 0
        np.maximum(phi, self.eps_phi, out=phi)
        np.minimum(phi, 1.0 - self.eps_phi, out=phi)
        self.EstParams.setField('phi', phi, dims=('K', 'D'))

    def updateEstParams_MAP(self, SS):
        ''' Update attribute EstParams for all comps given suff stats.

        Update uses the MAP objective for point estimation.

        Post Condition
        ---------
        Attributes K and EstParams updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'EstParams') or self.EstParams.K != SS.K:
            self.EstParams = ParamBag(K=SS.K, D=SS.D)
        phi_numer = SS.Count1 + self.Prior.lam1 - 1
        phi_denom = SS.Count1 + SS.Count0 + \
            self.Prior.lam1 + self.Prior.lam0 - 2
        phi = phi_numer / phi_denom
        self.EstParams.setField('phi', phi, dims=('K', 'D'))

    def updatePost(self, SS):
        ''' Update attribute Post for all comps given suff stats.

        Update uses the variational objective.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        self.ClearCache()
        if not hasattr(self, 'Post') or self.Post.K != SS.K:
            self.Post = ParamBag(K=SS.K, D=SS.D)

        lam1, lam0 = self.calcPostParams(SS)
        self.Post.setField('lam1', lam1, dims=self.CompDims + ('D',))
        self.Post.setField('lam0', lam0, dims=self.CompDims + ('D',))
        self.K = SS.K

    def calcPostParams(self, SS):
        ''' Calc posterior parameters for all comps given suff stats.

        Returns
        --------
        lam1 : 2D array, K x D (or K x K x D if relational)
        lam0 : 2D array, K x D (or K x K x D if relational)
        '''
        if SS.Count1.ndim == 2:
            lam1 = SS.Count1 + self.Prior.lam1[np.newaxis, :]
            lam0 = SS.Count0 + self.Prior.lam0[np.newaxis, :]
        elif SS.Count1.ndim == 3:
            lam1 = SS.Count1 + self.Prior.lam1[np.newaxis, np.newaxis, :]
            lam0 = SS.Count0 + self.Prior.lam0[np.newaxis, np.newaxis, :]
        return lam1, lam0

    def calcPostParamsForComp(self, SS, kA=None, kB=None):
        ''' Calc params (lam) for specific comp, given suff stats

            These params define the common-form of the exponential family
            Dirichlet posterior distribution over parameter vector phi

            Returns
            --------
            lam : 1D array, size D
        '''
        if kB is None:
            lam1_k = SS.Count1[kA].copy()
            lam0_k = SS.Count0[kA].copy()
        else:
            lam1_k = SS.Count1[kA] + SS.Count1[kB]
            lam0_k = SS.Count0[kA] + SS.Count0[kB]
        lam1_k += self.Prior.lam1
        lam0_k += self.Prior.lam0
        return lam1_k, lam0_k

    def updatePost_stochastic(self, SS, rho):
        ''' Update attribute Post for all comps given suff stats

        Update uses the stochastic variational formula.

        Post Condition
        ---------
        Attributes K and Post updated in-place.
        '''
        assert hasattr(self, 'Post')
        assert self.Post.K == SS.K
        self.ClearCache()

        lam1, lam0 = self.calcPostParams(SS)
        Post = self.Post
        Post.lam1[:] = (1 - rho) * Post.lam1 + rho * lam1
        Post.lam0[:] = (1 - rho) * Post.lam0 + rho * lam0

    def convertPostToNatural(self):
        ''' Convert current posterior params from common to natural form
        '''
        pass

    def convertPostToCommon(self):
        ''' Convert current posterior params from natural to common form
        '''
        pass

    def calcLogSoftEvMatrix_FromPost(self, Data, **kwargs):
        ''' Calculate expected log soft ev matrix under Post.

        Returns
        ------
        C : 2D array, size N x K
        '''
        # ElogphiT : vocab_size x K
        ElogphiT, Elog1mphiT = self.GetCached('E_logphiT_log1mphiT', 'all')
        return calcLogSoftEvMatrix_FromPost(
            Data,
            ElogphiT=ElogphiT,
            Elog1mphiT=Elog1mphiT,
            DataAtomType=self.DataAtomType, **kwargs)

    def calcELBO_Memoized(self, SS, returnVec=0, afterMStep=False, **kwargs):
        """ Calculate obsModel's objective using suff stats SS and Post.

        Args
        -------
        SS : bnpy SuffStatBag
        afterMStep : boolean flag
            if 1, elbo calculated assuming M-step just completed

        Returns
        -------
        obsELBO : scalar float
            Equal to E[ log p(x) + log p(phi) - log q(phi)]
        """
        Post = self.Post
        Prior = self.Prior
        if not afterMStep:
            ElogphiT = self.GetCached('E_logphiT', 'all')
            Elog1mphiT = self.GetCached('E_log1mphiT', 'all')
            # with relational/graph datasets, these have shape D x K x K
            # otherwise, these have shape D x K
        if self.CompDims == ('K',):
            # Typical case: K x D
            assert Post._FieldDims['lam1'] == ('K', 'D')
            L_perComp = np.zeros(SS.K)
            for k in range(SS.K):
                L_perComp[k] = c_Diff(Prior.lam1, Prior.lam0,
                                      Post.lam1[k], Post.lam0[k])
                if not afterMStep:
                    L_perComp[k] += np.inner(
                        SS.Count1[k] + Prior.lam1 - Post.lam1[k],
                        ElogphiT[:, k])
                    L_perComp[k] += np.inner(
                        SS.Count0[k] + Prior.lam0 - Post.lam0[k],
                        Elog1mphiT[:, k])
            if returnVec:
                return L_perComp
            return np.sum(L_perComp)
        elif self.CompDims == ('K','K',):
            # Relational case, K x K x D
            assert Post._FieldDims['lam1'] == ('K', 'K', 'D')

            cPrior = c_Func(Prior.lam1, Prior.lam0)
            Ldata = SS.K * SS.K * cPrior - c_Func(Post.lam1, Post.lam0)
            if not afterMStep:
                Ldata += np.sum(
                    (SS.Count1 + Prior.lam1[nx, nx, :] - Post.lam1) *
                    ElogphiT.T)
                Ldata += np.sum(
                    (SS.Count0 + Prior.lam0[nx, nx, :] - Post.lam0) *
                    Elog1mphiT.T)
            return Ldata
        else:
            raise ValueError("Unrecognized compdims: " + str(self.CompDims))

    def getDatasetScale(self, SS, extraSS=None):
        ''' Get number of observed scalars in dataset from suff stats.

        Used for normalizing the ELBO so it has reasonable range.

        Returns
        ---------
        s : scalar positive integer
        '''
        s = SS.Count1.sum() + SS.Count0.sum()
        if extraSS is None:
            return s
        else:
            sextra = extraSS.Count1.sum() + extraSS.Count0.sum()
            return s - sextra

    def calcHardMergeGap(self, SS, kA, kB):
        ''' Calculate change in ELBO after a hard merge applied to this model

        Returns
        ---------
        gap : scalar real, indicates change in ELBO after merge of kA, kB
        '''
        Prior = self.Prior
        cPrior = c_Func(Prior.lam1, Prior.lam0)

        Post = self.Post
        cA = c_Func(Post.lam1[kA], Post.lam0[kA])
        cB = c_Func(Post.lam1[kB], Post.lam0[kB])

        lam1, lam0 = self.calcPostParamsForComp(SS, kA, kB)
        cAB = c_Func(lam1, lam0)
        return cA + cB - cPrior - cAB

    def calcHardMergeGap_AllPairs(self, SS):
        ''' Calculate change in ELBO for all candidate hard merge pairs

        Returns
        ---------
        Gap : 2D array, size K x K, upper-triangular entries non-zero
              Gap[j,k] : scalar change in ELBO after merge of k into j
        '''
        Prior = self.Prior
        cPrior = c_Func(Prior.lam1, Prior.lam0)

        Post = self.Post
        c = np.zeros(SS.K)
        for k in range(SS.K):
            c[k] = c_Func(Post.lam1[k], Post.lam0[k])

        Gap = np.zeros((SS.K, SS.K))
        for j in range(SS.K):
            for k in range(j + 1, SS.K):
                cjk = c_Func(*self.calcPostParamsForComp(SS, j, k))
                Gap[j, k] = c[j] + c[k] - cPrior - cjk
        return Gap

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

    def calcHardMergeGap_SpecificPairSS(self, SS1, SS2):
        ''' Calc change in ELBO for merge of two K=1 suff stat bags.

        Returns
        -------
        gap : scalar float
        '''
        assert SS1.K == 1
        assert SS2.K == 1

        Prior = self.Prior
        cPrior = c_Func(Prior.lam1, Prior.lam0)

        # Compute cumulants of individual states 1 and 2
        lam11, lam10 = self.calcPostParamsForComp(SS1, 0)
        lam21, lam20 = self.calcPostParamsForComp(SS2, 0)
        c1 = c_Func(lam11, lam10)
        c2 = c_Func(lam21, lam20)

        # Compute cumulant of merged state 1&2
        SSM = SS1 + SS2
        lamM1, lamM0 = self.calcPostParamsForComp(SSM, 0)
        cM = c_Func(lamM1, lamM0)

        return c1 + c2 - cPrior - cM

    def calcLogMargLikForComp(self, SS, kA, kB=None, **kwargs):
        ''' Calc log marginal likelihood of data assigned to given component

        Args
        -------
        SS : bnpy suff stats object
        kA : integer ID of target component to compute likelihood for
        kB : (optional) integer ID of second component.
             If provided, we merge kA, kB into one component for calculation.
        Returns
        -------
        logM : scalar real
               logM = log p( data assigned to comp kA )
                      computed up to an additive constant
        '''
        return -1 * c_Func(*self.calcPostParamsForComp(SS, kA, kB))

    def calcMargLik(self, SS):
        ''' Calc log marginal likelihood combining all comps, given suff stats

        Returns
        --------
        logM : scalar real
               logM = \sum_{k=1}^K log p( data assigned to comp k | Prior)
        '''
        return self.calcMargLik_CFuncForLoop(SS)

    def calcMargLik_CFuncForLoop(self, SS):
        Prior = self.Prior
        logp = np.zeros(SS.K)
        for k in range(SS.K):
            lam1, lam0 = self.calcPostParamsForComp(SS, k)
            logp[k] = c_Diff(Prior.lam1, Prior.lam0,
                             lam1, lam0)
        return np.sum(logp)

    def _E_logphi(self, k=None):
        if k is None or k == 'prior':
            lam1 = self.Prior.lam1
            lam0 = self.Prior.lam0
        elif k == 'all':
            lam1 = self.Post.lam1
            lam0 = self.Post.lam0
        else:
            lam1 = self.Post.lam1[k]
            lam0 = self.Post.lam0[k]
        Elogphi = digamma(lam1) - digamma(lam1 + lam0)
        return Elogphi

    def _E_log1mphi(self, k=None):
        if k is None or k == 'prior':
            lam1 = self.Prior.lam1
            lam0 = self.Prior.lam0
        elif k == 'all':
            lam1 = self.Post.lam1
            lam0 = self.Post.lam0
        else:
            lam1 = self.Post.lam1[k]
            lam0 = self.Post.lam0[k]
        Elog1mphi = digamma(lam0) - digamma(lam1 + lam0)
        return Elog1mphi

    def _E_logphiT_log1mphiT(self, k=None):
        if k == 'all':
            lam1T = self.Post.lam1.T.copy()
            lam0T = self.Post.lam0.T.copy()
            digammaBoth = digamma(lam1T + lam0T)
            ElogphiT = digamma(lam1T) - digammaBoth
            Elog1mphiT = digamma(lam0T) - digammaBoth
        else:
            ElogphiT = self._E_logphiT(k)
            Elog1mphiT = self._E_log1mphiT(k)
        return ElogphiT, Elog1mphiT

    def _E_logphiT(self, k=None):
        ''' Calculate transpose of expected phi matrix

        Important to make a copy of the matrix so it is C-contiguous,
        which leads to much much faster matrix operations.

        Returns
        -------
        ElogphiT : 2D array, vocab_size x K
        '''
        if k == 'all':
            dlam1T = self.Post.lam1.T.copy()
            dlambothT = self.Post.lam0.T.copy()
            dlambothT += dlam1T
            digamma(dlam1T, out=dlam1T)
            digamma(dlambothT, out=dlambothT)
            return dlam1T - dlambothT
        ElogphiT = self._E_logphi(k).T.copy()
        return ElogphiT

    def _E_log1mphiT(self, k=None):
        ''' Calculate transpose of expected 1-minus-phi matrix

        Important to make a copy of the matrix so it is C-contiguous,
        which leads to much much faster matrix operations.

        Returns
        -------
        ElogphiT : 2D array, vocab_size x K
        '''
        if k == 'all':
            # Copy so lam1T/lam0T are C-contig and can be shared mem.
            lam1T = self.Post.lam1.T.copy()
            lam0T = self.Post.lam0.T.copy()
            return digamma(lam0T) - digamma(lam1T + lam0T)

        ElogphiT = self._E_log1mphi(k).T.copy()
        return ElogphiT

    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType,
                    DataAtomType=self.DataAtomType,
                    K=self.K)

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step.

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        ElogphiT, Elog1mphiT = self.GetCached('E_logphiT_log1mphiT', 'all')
        K = self.K
        if ShMem is None:
            ShMem = dict()
        if 'ElogphiT' not in ShMem:
            ShMem['ElogphiT'] = numpyToSharedMemArray(ElogphiT)
            ShMem['Elog1mphiT'] = numpyToSharedMemArray(Elog1mphiT)
        else:
            ElogphiT_shView = sharedMemToNumpyArray(ShMem['ElogphiT'])
            assert ElogphiT_shView.shape >= K
            ElogphiT_shView[:, :K] = ElogphiT

            Elog1mphiT_shView = sharedMemToNumpyArray(ShMem['Elog1mphiT'])
            assert Elog1mphiT_shView.shape >= K
            Elog1mphiT_shView[:, :K] = Elog1mphiT
        return ShMem

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return calcLocalParams, calcSummaryStats


    def calcSmoothedMu(self, X, W=None):
        ''' Compute smoothed estimate of probability of each word.

        Returns
        -------
        Mu : 1D array, size D (aka vocab_size)
        '''
        if X is None:
            return self.Prior.lam1 / (self.Prior.lam1 + self.Prior.lam0)

        if X.ndim > 1:
            if W is None:
                NX = X.shape[0]
                X = np.sum(X, axis=0)
            else:
                NX = np.sum(W)
                X = np.dot(W, X)
        else:
            NX = 1

        assert X.ndim == 1
        assert X.size == self.D
        Mu = X + self.Prior.lam1
        Mu /= (NX + self.Prior.lam1 + self.Prior.lam0)
        return Mu

    def calcSmoothedBregDiv(self,
            X, Mu, W=None,
            smoothFrac=0.0,
            includeOnlyFastTerms=False,
            DivDataVec=None,
            returnDivDataVec=False,
            return1D=False,
            **kwargs):
        ''' Compute Bregman divergence between data X and clusters Mu.

        Smooth the data via update with prior parameters.

        Returns
        -------
        Div : 2D array, N x K
            Div[n,k] = smoothed distance between X[n] and Mu[k]
        '''
        if X.ndim < 2:
            X = X[np.newaxis,:]
        assert X.ndim == 2
        N = X.shape[0]
        D = X.shape[1]

        if W is not None:
            assert W.ndim == 1
            assert W.size == N

        if not isinstance(Mu, list):
            Mu = (Mu,)
        K = len(Mu)
        assert Mu[0].size == D

        # Smooth-ify the data matrix X
        if smoothFrac == 0:
            MuX = np.minimum(X, 1 - 1e-14)
            np.maximum(MuX, 1e-14, out=MuX)
        else:
            MuX = X + smoothFrac * self.Prior.lam1
            NX = 1.0 + smoothFrac * (self.Prior.lam1 + self.Prior.lam0)
            MuX /= NX

        # Compute Div array up to a per-row additive constant indep. of k
        Div = np.zeros((N, K))
        for k in range(K):
            Div[:,k] = -1 * np.sum(MuX * np.log(Mu[k]), axis=1) + \
                -1 * np.sum((1-MuX) * np.log(1-Mu[k]), axis=1)

        if not includeOnlyFastTerms:
            if DivDataVec is None:
                # Compute DivDataVec : 1D array of size N
                # This is the per-row additive constant indep. of k.

                # STEP 1: Compute MuX * log(MuX)
                logMuX = np.log(MuX)
                MuXlogMuX = logMuX
                MuXlogMuX *= MuX
                DivDataVec = np.sum(MuXlogMuX, axis=1)

                # STEP 2: Compute (1-MuX) * log(1-MuX)
                OneMinusMuX = MuX
                OneMinusMuX *= -1
                OneMinusMuX += 1
                logOneMinusMuX = logMuX
                np.log(OneMinusMuX, out=logOneMinusMuX)
                logOneMinusMuX *= OneMinusMuX
                DivDataVec += np.sum(logOneMinusMuX, axis=1)
            Div += DivDataVec[:,np.newaxis]

        assert np.all(np.isfinite(Div))
        # Apply per-atom weights to divergences.
        if W is not None:
            assert W.ndim == 1
            assert W.size == N
            Div *= W[:,np.newaxis]
        # Verify divergences are strictly non-negative
        if not includeOnlyFastTerms:
            minDiv = Div.min()
            if minDiv < 0:
                if minDiv < -1e-6:
                    raise AssertionError(
                        "Expected Div.min() to be positive or" + \
                        " indistinguishable from zero. Instead " + \
                        " minDiv=% .3e" % (minDiv))
                np.maximum(Div, 0, out=Div)
                minDiv = Div.min()
            assert minDiv >= 0

        if return1D:
            Div = Div[:,0]
        if returnDivDataVec:
            return Div, DivDataVec
        return Div

        ''' OLD VERSION
        if smoothFrac == 0:
            MuX = np.minimum(X, 1 - 1e-14)
            MuX = np.maximum(MuX, 1e-14)
        else:
            MuX = X + smoothFrac * self.Prior.lam1
            NX = 1.0 + smoothFrac * (self.Prior.lam1 + self.Prior.lam0)
            MuX /= NX

        Div = np.zeros((N, K))
        for k in xrange(K):
            Mu_k = Mu[k][np.newaxis,:]
            Div[:,k] = np.sum(MuX * np.log(MuX / Mu_k), axis=1) + \
                np.sum((1-MuX) * np.log((1-MuX) / (1-Mu_k)), axis=1)
        '''

    def calcBregDivFromPrior(self, Mu, smoothFrac=0.0):
        ''' Compute Bregman divergence between Mu and prior mean.

        Returns
        -------
        Div : 1D array, size K
            Div[k] = distance between Mu[k] and priorMu
        '''
        if not isinstance(Mu, list):
            Mu = (Mu,) # cheaper than a list
        K = len(Mu)

        priorMu = self.Prior.lam1 / (self.Prior.lam1 + self.Prior.lam0)
        priorN = (1-smoothFrac) * (self.Prior.lam1 + self.Prior.lam0)

        Div = np.zeros((K, self.D))
        for k in range(K):
            Div[k, :] = priorMu * np.log(priorMu / Mu[k]) + \
                (1-priorMu) * np.log((1-priorMu)/(1-Mu[k]))
        return np.dot(Div, priorN)



def c_Func(lam1, lam0):
    ''' Evaluate cumulant function at given params.

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    assert lam1.ndim == lam0.ndim
    return np.sum(gammaln(lam1 + lam0) - gammaln(lam1) - gammaln(lam0))


def c_Diff(lamA1, lamA0, lamB1, lamB0):
    ''' Evaluate difference of cumulant functions c(params1) - c(params2)

    May be more numerically stable than directly using c_Func
    to find the difference.

    Returns
    -------
    diff : scalar real value of the difference in cumulant functions
    '''
    return c_Func(lamA1, lamA0) - c_Func(lamB1, lamB0)


def calcLocalParams(Dslice, **kwargs):
    """ Calculate local parameters for provided slice of data.

    Returns
    -------
    LP : dict with fields
        * E_log_soft_ev : 2D array, size N x K
    """
    E_log_soft_ev = calcLogSoftEvMatrix_FromPost(Dslice, **kwargs)
    return dict(E_log_soft_ev=E_log_soft_ev)


def calcLogSoftEvMatrix_FromPost(Data,
                                 ElogphiT=None,
                                 Elog1mphiT=None,
                                 K=None,
                                 DataAtomType='doc',
                                 **kwargs):
    ''' Calculate expected log soft ev matrix.

    Model Args
    ------
    ElogphiT : vocab_size x K matrix

    Data Args
    ---------
    Data : bnpy Data object

    Returns
    ------
    L : 2D array, size N x K
    '''
    if K is None:
        K = ElogphiT.shape[1]
    if hasattr(Data, 'X'):
        if ElogphiT.ndim == 2:
            # Typical case
            C = np.dot(Data.X, ElogphiT[:, :K]) + \
                np.dot(1.0 - Data.X, Elog1mphiT[:, :K])
        else:
            # Relational case
            C = np.tensordot(Data.X, ElogphiT[:, :K, :K], axes=1) + \
                np.tensordot(1.0 - Data.X, Elog1mphiT[:, :K, :K], axes=1)
    else:
        assert Elog1mphiT.ndim == 2
        if DataAtomType == 'doc':
            X = Data.getSparseDocTypeBinaryMatrix()
            C = X * ElogphiT
            C_1mX = Elog1mphiT.sum(axis=0)[np.newaxis, :] - X * Elog1mphiT
            C += C_1mX
        else:
            C = np.tile(Elog1mphiT, (Data.nDoc, 1))
            for d in range(Data.nDoc):
                start_d = Data.vocab_size * d
                words_d = Data.word_id[
                    Data.doc_range[d]:Data.doc_range[d+1]]
                C[start_d + words_d, :] = ElogphiT[words_d, :]
    return C

def calcSummaryStats(Dslice, SS, LP, DataAtomType='doc', **kwargs):
    ''' Calculate summary statistics for given dataset and local parameters

    Returns
    --------
    SS : SuffStatBag object, with K components.
    '''
    if 'resp' in LP:
        N = LP['resp'].shape[0]
        K = LP['resp'].shape[1]
        if LP['resp'].ndim == 2:
            CompDims = ('K',)  # typical case
        else:
            assert LP['resp'].ndim == 3
            CompDims = ('K', 'K')  # relational data
    else:
        assert 'spR' in LP
        N, K = LP['spR'].shape
        CompDims = ('K',)

    if SS is None:
        SS = SuffStatBag(K=K, D=Dslice.dim)
    if not hasattr(SS, 'N'):
        if 'resp' in LP:
            SS.setField('N', np.sum(LP['resp'], axis=0), dims=CompDims)
        else:
            SS.setField('N', LP['spR'].sum(axis=0), dims=CompDims)

    if hasattr(Dslice, 'X'):
        X = Dslice.X
        if 'resp' in LP:
            # Matrix-matrix product, result is K x D (or KxKxD if relational)
            CountON = np.tensordot(LP['resp'].T, X, axes=1)
            CountOFF = np.tensordot(LP['resp'].T, 1 - X, axes=1)
        else:
            CountON = LP['spR'].T * X
            CountOFF = LP['spR'].T * (1 - X)
    elif DataAtomType == 'doc' or Dslice.nDoc == N:
        X = Dslice.getSparseDocTypeBinaryMatrix()
        if 'resp' in LP:
            # Sparse matrix product
            CountON = LP['resp'].T * X
        else:
            CountON = (LP['spR'].T * X).toarray()
        CountOFF = SS.N[:, np.newaxis] - CountON
    else:
        CountON = np.zeros((SS.K, Dslice.vocab_size))
        CountOFF = np.zeros((SS.K, Dslice.vocab_size))
        for d in range(Dslice.nDoc):
            words_d = Dslice.word_id[
                Dslice.doc_range[d]:Dslice.doc_range[d+1]]
            rstart_d = d * Dslice.vocab_size
            rstop_d = (d+1) * Dslice.vocab_size
            if 'resp' in LP:
                Count_d = Resp[rstart_d:rstop_d, :].T
            else:
                raise NotImplementedError("TODO")
            CountOFF += Count_d
            CountON[:, words_d] += Count_d[:, words_d]
            CountOFF[:, words_d] -= Count_d[:, words_d]
    SS.setField('Count1', CountON, dims=CompDims + ('D',))
    SS.setField('Count0', CountOFF, dims=CompDims + ('D',))
    return SS
