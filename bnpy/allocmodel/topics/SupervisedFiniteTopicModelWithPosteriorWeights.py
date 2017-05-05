import numpy as np
import math
from numpy.linalg import inv

import itertools

from ..AllocModel import AllocModel
from bnpy.suffstats import ParamBag, SuffStatBag
from ...util import digamma, gammaln
from ...util import NumericUtil

from bnpy.util import LOGTWO, LOGPI, LOGTWOPI, EPS
from bnpy.util import dotATA, dotATB, dotABT
from bnpy.util import as1D, as2D, as3D, toCArray, np2flatstr

from SupervisedLocalStepManyDocs2 import calcLocalParams


class SupervisedFiniteTopicModel3(AllocModel):

    '''
    Bayesian topic model with a finite number of components K.

    Attributes
    -------
    inferType : string {'VB', 'moVB', 'soVB'}
            indicates which updates to perform for local/global steps
    K : int
            number of components
    alpha : float
            scalar pseudo-count
            used in Dirichlet prior on document -topic probabilities \pi_d.


    Attributes for VB
    ---------
    None. No global structure exists except scalar parameter gamma.

    Variational Local Parameters
    --------
    resp :  2D array, N x K
            q(z_n) = Categorical( resp_{n1}, ... resp_{nK} )
    theta : 2D array, nDoc x K
            q(pi_d) = Dirichlet( \theta_{d1}, ... \theta_{dK} )

    References
    -------
    Supervised Latent Dirichlet Allocation, by McAulliffe and Blei
    introduces a supervised topic model with Dirichlet-Mult observations.
    '''

    def __init__(self, inferType, priorDict=None):
        #print 'Init SupervisedFiniteTopicModel3'

        if inferType == 'EM':
            raise ValueError('SupervisedFiniteTopicModel cannot do EM.')
        self.inferType = inferType
        self.K = 0

        if priorDict is None:
            self.set_prior()
        else:
            self.set_prior(**priorDict)

        self.Prior.mean_weights = self.Prior.w_J
        self.mean_weights = self.Prior.w_J
        self.Prior.stdev = self.Prior.ptau / (self.Prior.pnu - 2)
        self.stdev = self.Prior.ptau / (self.Prior.pnu - 2)
        #print self.mean_weights

    def get_active_comp_probs(self):
        ''' Get vector of appearance probabilities for each active comp.

        Returns
        -------
        beta : 1D array, size K
                beta[k] gives probability of comp. k under this model.
        '''
        return np.ones(self.K) / float(self.K)

    def set_prior(self, alpha=1.0, **kwargs):
        #print 'Set Prior'
        self.alpha = float(alpha)

        try:
            self.Prior = calcPrior(K=self.K,**kwargs)
        except:
            self.Prior = calcPrior(**kwargs)

        self.Prior.setField('alpha', alpha, dims=None)

        self.Prior.setField('mean_weights', self.Prior.w_J, dims='J')
        self.Prior.setField('stdev', self.Prior.ptau / (self.Prior.pnu - 2), dims=None)
        #print self.mean_weights

    def to_dict(self):
        return dict(mean_weights=self.mean_weights,
                                stdev=self.stdev,
                                pnu=self.Prior.pnu,
                                ptau=self.Prior.ptau,
                                w_J=self.Prior.w_J,
                                P_JJ=self.Prior.P_JJ,
                                Pw_J=self.Prior.Pw_J,
                                wPw_1=self.Prior.wPw_1,
                                K=self.K)

    def from_dict(self, Dict):
        self.inferType = Dict['inferType']
        self.K = Dict['K']
        self.mean_weights = Dict['mean_weights']
        self.stdev = Dict['stdev']
        self.Prior.pnu = Dict['pnu']
        self.Prior.ptau = Dict['ptau']
        self.Prior.w_J = Dict['w_J']
        self.Prior.P_JJ = Dict['P_JJ']
        self.Prior.Pw_J = Dict['Pw_J']
        self.Prior.wPw_1 = Dict['wPw_1']

    def get_prior_dict(self):
        return dict(alpha=self.Prior.alpha,
                                K=self.K,
                                inferType=self.inferType,
                                mean_weights=self.Prior.mean_weights,
                                stdev=self.Prior.stdev,
                                pnu=self.Prior.pnu,
                                ptau=self.Prior.ptau,
                                w_J=self.Prior.w_J,
                                P_JJ=self.Prior.P_JJ,
                                Pw_J=self.Prior.Pw_J,
                                wPw_1=self.Prior.wPw_1)

    def get_info_string(self):
        ''' Returns human-readable name of this object
        '''
        return 'Supervised Finite LDA model with K=%d comps. alpha=%.2f' \
                % (self.K, self.Prior.alpha)

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for each data item.

        Parameters
        -------
        Data : bnpy.data.DataObj subclass

        LP : dict
                Local parameters as key-value string/array pairs
                * E_log_soft_ev : 2D array, N x K
                        E_log_soft_ev[n,k] = log p(data obs n | comp k)

        Returns
        -------
        LP : dict
                Local parameters, with updated fields
                * resp : 2D array, N x K
                        Posterior responsibility each comp has for each item
                        resp[n, k] = p(z[n] = k | x[n])
                * theta : 2D array, nDoc x K
                        Defines approximate posterior on doc-topic weights.
                        q(\pi_d) = Dirichlet(theta[d,0], ... theta[d, K-1])
        '''
        print 'Calc Local Params with....'
        print 'cur weights: ', np.round(self.mean_weights,2)
        print 'cur stdev: ', np.round(self.stdev,4)

        LP = calcLocalParams(
                Data, LP, eta=self.mean_weights, alpha=self.alpha, delta=self.stdev, **kwargs)
            #Data, LP, eta=self.mean_weights, alpha=self.alpha, delta_stdev=self.stdev, **kwargs)
        assert 'resp' in LP
        assert 'theta' in LP
        assert 'DocTopicCount' in LP

        return LP

    def initLPFromResp(self, Data, LP):
        ''' Fill in remaining local parameters given token-topic resp.

        Args
        ----
        LP : dict with fields
                * resp : 2D array, size N x K

        Returns
        -------
        LP : dict with fields
                * DocTopicCount
                * theta
                * ElogPi
        '''

        resp = LP['resp']
        K = resp.shape[1]
        DocTopicCount = np.zeros((Data.nDoc, K))
        resp_idx_start = 0
        for d in xrange(Data.nDoc):
            start = Data.doc_range[d]
            stop = Data.doc_range[d + 1]
            DocTopicCount[d, :] = np.sum(resp[start:stop, :], axis=0)
        remMass = np.minimum(0.1, 1.0 / (K * K))
        newEbeta = (1 - remMass) / K
        theta = DocTopicCount + self.alpha * newEbeta
        digammaSumTheta = digamma(theta.sum(axis=1))
        ElogPi = digamma(theta) - digammaSumTheta[:, np.newaxis]

        LP['DocTopicCount'] = DocTopicCount
        LP['theta'] = theta
        LP['ElogPi'] = ElogPi

        return LP

    def get_global_suff_stats(self, Data, LP,
                                                      doPrecompEntropy=None,
                                                      cslice=(0, None), **kwargs):
        ''' Calculate sufficient statistics for global updates.

        Parameters
        -------
        Data : bnpy data object
        LP : local param dict with fields
                resp : Data.nObs x K array,
                        where resp[n,k] = posterior resp of comp k
        doPrecompEntropy : boolean flag
                indicates whether to precompute ELBO terms in advance
                used for memoized learning algorithms (moVB)

        Returns
        -------
        SS : SuffStatBag with K components
                Relevant fields
                * nDoc : scalar float
                        Counts total documents available in provided data.

                Also has optional ELBO field when precompELBO is True
                * Hvec : 1D array, size K
                        Vector of entropy contributions from each comp.
                        Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
        '''

        #print 'Get Global Suff Stats'
        resp = LP['resp']
        nTokensTotal, K = resp.shape

        SS = SuffStatBag(K=K, J=K, D=Data.dim)

        SS = calcSummaryStats(Data,SS,LP,**kwargs)

        if cslice[1] is None:
            SS.setField('nDoc', Data.nDoc, dims=None)
        else:
            SS.setField('nDoc', cslice[1] - cslice[0], dims=None)

        if doPrecompEntropy:
            Hvec = self.L_entropy(Data, LP, returnVector=1)
            Lalloc = self.L_alloc(Data, LP)
            #Lslda = self.L_supervised(Data, LP)
            LGaussianReg = self.L_GaussianReg(Data,LP)

            SS.setELBOTerm('Hvec', Hvec, dims='K')
            SS.setELBOTerm('L_alloc', Lalloc, dims=None)
            #SS.setELBOTerm('L_sLDA', Lslda)
            SS.setELBOTerm('L_GaussianRegresion', LGaussianReg)

        return SS

    def update_global_params(self, SS, rho=None, **kwargs):
        ''' Update global parameters to optimize the ELBO objective.
        '''
        #print 'Update Global Params'
        self.Post =     update_global_params(SS,self.Prior)

        self.mean_weights = self.Post.mean_weights
        self.stdev = self.Post.stdev
        #print self.mean_weights

    def set_global_params(self, K=0, mean_weights=None, delta_stdev=None, **kwargs):
        """ Set global parameters to provided values.
        """
        '''
        if eta is not None:
                self.eta = eta
        if delta_stdev is not None:
                self.delta_stdev = delta_stdev
        '''
        print 'Set Global Params'
        '''if mean_weights is not None:
                self.mean_weights = mean_weights
        else:
                self.mean_weights = self.Prior.w_J
        if delta_stdev is not None:
                self.stdev = delta_stdev
        else:
                self.stdev = self.Prior.ptau / (self.Prior.pnu - 2)
        '''

        self.K = K
        #print self.Prior.pnu
        #print self.Prior.w_J

        self.set_prior(pnu=self.Prior.pnu,ptau=self.Prior.ptau,K=self.K,**kwargs)


        self.mean_weights = self.Prior.w_J
        self.stdev = self.Prior.ptau / (self.Prior.pnu - 2)
        #print self.mean_weights


        '''
        if hmodel is not None:
                self.setParamsFromHModel(hmodel)
        else:
                raise ValueError("Unrecognized set_global_params args")
        '''

    def init_global_params(self, Data, K=0, mean_weights=None, delta_stdev=None, **kwargs):
        """ Initialize global parameters to provided values.
        """
        #print 'Init Global Params'
        self.K = K
        #if priorDict is None:
        #               self.set_prior()
        #else:
        #self.set_prior(**kwargs)
        self.set_prior(pnu=self.Prior.pnu,ptau=self.Prior.ptau,K=self.K,**kwargs)

        #print self.Prior.pnu
        #print self.Prior.ptau
        self.mean_weights = self.Prior.w_J
        self.stdev = self.Prior.ptau / (self.Prior.pnu - 2)
        #print self.mean_weights

        '''
        if mean_weights is None:
                self.mean_weights = np.ones(K)
        else:
                print 'init mean weights to  param'
                self.mean_weights = mean_weights
        if delta_stdev is None:
                self.stdev = 0.1
        else:
                self.stdev = delta_stdev
        '''
    def setParamsFromHModel(self, hmodel):
        """ Set parameters exactly as in provided HModel object.

        Parameters
        ------
        hmodel : bnpy.HModel
                The model to copy parameters from.

        Post Condition
        ------
        Attributes rho/omega set exactly equal to hmodel's allocModel.
        """
        print 'Set Params from HModel'
        self.K = hmodel.allocModel.K
        if hasattr(hmodel.allocModel, 'eta'):
            eta = hmodel.allocModel.eta.copy()
        elif hasattr(hmodel.allocModel,'mean_weights'):
            mean_weights = hmodel.allocModel.mean_weights.copy()
        elif hasattr(hmodel.allocmodel, 'stdev'):
            stdev = hmodel.allocModel.stdev.copy()
        else:
            raise AttributeError('Unrecognized hmodel')

    def calc_evidence(self, Data, SS, LP, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Returns
        -------
        L : float
                Represents sum of all terms in ELBO objective.
        """
        #print 'Calc Ev'
        if SS.hasELBOTerms():
            Lentropy = SS.getELBOTerm('Hvec').sum()
            Lalloc = SS.getELBOTerm('L_alloc')
            #Lsupervised = SS.getELBOTerm('L_supervised')
            LGaussianRegression = SS.getELBOTerm('L_GaussianReg')
        else:
            Lentropy = self.L_entropy(Data, LP, returnVector=0)
            Lalloc = self.L_alloc(Data, LP)
            #Lsupervised = self.L_supervised(Data, LP)
            LGaussianReg = self.L_GaussianReg(Data,SS,LP)
        if SS.hasAmpFactor():
            Lentropy *= SS.ampF
            Lalloc *= SS.ampF
            #Lsupervised *= SS.ampF
            LGaussianReg *= SS.ampF

        return Lalloc + Lentropy + LGaussianReg

    def L_entropy(self, Data, LP, returnVector=1):
        ''' Calculate assignment entropy term of the ELBO objective.

        Returns
        -------
        Hvec : 1D array, size K
                Hvec[k] = \sum_{n=1}^N H[q(z_n)]
        '''
        return L_entropy(Data, LP, returnVector=returnVector)

    def L_alloc(self, Data, LP):
        ''' Calculate allocation term of the ELBO objective.

        Returns
        -------
        L_alloc : scalar float
        '''
        return L_alloc(Data=Data, LP=LP, alpha=self.alpha)

    def L_supervised(self, Data, LP):
        ''' Calculate supervised term of the ELBO objective.

        Returns
        -------
        L_supervised : scalar float
        '''
        return L_supervised(Data=Data, LP=LP, eta=self.Post.mean_weights, delta=self.Post.stdev)

    def L_GaussianReg(self, Data, SS, LP):
        ''' Calculate supervised term of the ELBO objective.

        Returns
        -------
        L_supervised : scalar float
        '''
        return L_GaussianReg(Data=Data, SS=SS, LP=LP,Prior=self.Prior,Post=self.Post)


    def getSerializableParamsForLocalStep(self):
        """ Get compact dict of params for parallel local step.

        Returns
        -------
        Info : dict
        """
        return dict(inferType=self.inferType,
                                K=self.K,
                                alpha=self.alpha,
                                delta_stdev=self.delta_stdev,
                                eta=self.eta)

    def fillSharedMemDictForLocalStep(self, ShMem=None):
        """ Get dict of shared mem arrays needed for parallel local step.

        Returns
        -------
        ShMem : dict of RawArray objects
        """
        # No shared memory required here.
        if isinstance(ShMem, dict):
            return ShMem
        else:
            return dict()

    def getLocalAndSummaryFunctionHandles(self):
        """ Get function handles for local step and summary step

        Useful for parallelized algorithms.

        Returns
        -------
        calcLocalParams : f handle
        calcSummaryStats : f handle
        """
        return calcLocalParams, calcSummaryStats


def L_alloc(Data=None, LP=None, nDoc=0, alpha=1.0, **kwargs):
    ''' Calculate allocation term of the ELBO objective.

    E[ log p(pi) + log p(z) - log q(pi)  ]

    Returns
    -------
    L_alloc : scalar float
    '''

    if Data is not None:
        nDoc = Data.nDoc
    if LP is None:
        LP = dict(**kwargs)

    K = LP['DocTopicCount'].shape[1]
    cDiff = nDoc * c_Func(alpha, K) - c_Func(LP['theta'])
    slackVec = LP['DocTopicCount'] + alpha - LP['theta']
    slackVec *= LP['ElogPi']

    return cDiff + np.sum(slackVec)


def L_entropy(Data=None, LP=None, resp=None, returnVector=0):
    """ Calculate entropy of soft assignments term in ELBO objective.

    Returns
    -------
    L_entropy : scalar float
    """

    if LP is not None:
        resp = LP['resp']

    if hasattr(Data, 'word_count') and resp.shape[0] == Data.nUniqueToken:
        Hvec = -1 * NumericUtil.calcRlogRdotv(resp, Data.word_count)
    else:
        Hvec = -1 * NumericUtil.calcRlogR(resp)

    assert Hvec.min() >= -1e-6
    if returnVector:
        return Hvec

    return Hvec.sum()


def L_supervised(Data=None, LP=None, eta=None, delta=None):
    """Calculate slda term of the ELBO objective.

    E[p(y)]

    Returns
    -------
    L_supervised : scalar float
    """
    if LP is not None:
        resp = LP['resp']

    response = Data.response
    L_supervised = 0
    nDoc = Data.nDoc

    for d in xrange(nDoc):
        start = Data.doc_range[d]
        stop = Data.doc_range[d+1]

        wc_d = Data.word_count[start:stop]
        N_d = sum(wc_d)
        resp_d = np.asarray(resp[start:stop, :])

        response_d = response[d]
        nTokens_d, K = resp_d.shape

        weighted_resp_d = wc_d[:, None] * resp_d
        EZ_d = np.sum(weighted_resp_d, axis=0) / float(N_d)

        EZTZ_d = calc_EzzT_one_doc_gaussianreg(wc_d, resp_d)

        sterm = np.dot(eta, EZTZ_d)
        sterm = np.dot(sterm, eta)

        L_supervised_d = (-0.5) * np.log(2.0 * math.pi * delta)
        L_supervised_d -= np.square(response_d) / (2.0 * delta)
        L_supervised_d += (response_d / delta) * np.inner(eta, EZ_d)
        L_supervised_d -= sterm / (2.0 * delta)

        L_supervised += L_supervised_d

    return L_supervised


def L_GaussianReg(Data=None,LP=None,SS=None,Prior=None,Post=None,
        **kwargs):
    ''' Calculate expected log soft ev matrix under approximate posterior

    Returns
    -------
    L_GaussianRegression
    '''
    K = SS.K
    J = SS.K

    #E_log_d = digamma(0.5 * Prior.pnu) - np.log(0.5 * Prior.ptau)
    #E_d = (Prior.pnu / Prior.ptau)
    E_log_d = digamma(0.5 * Post.pnu) - np.log(0.5 * Post.ptau)
    E_d = (Post.pnu / Post.ptau)

    if 'resp' not in LP:
        resp = np.ones((int(Data.nUniqueToken), K)) * (1/K)
    else:
        resp = LP['resp']

    Y_D = Data.response

    Prior_cFunc = cFunc_logreg(pnu=Prior.pnu, ptau=Prior.ptau, w_J=Prior.w_J, P_JJ=Prior.P_JJ)
    Post_cFunc = cFunc_logreg(pnu=Post.pnu, ptau=Post.ptau, w_J=Post.w_J, P_JJ=Post.P_JJ)

    #E_logdw = E_log_d * Prior.w_J
    #E_logdwTw = np.linalg.inv(Prior.P_JJ) + (Prior.pnu / Prior.ptau) * np.dot(Prior.w_J,Prior.w_J)
    E_logdw = E_log_d * Post.w_J
    E_logdwTw = np.linalg.inv(Post.P_JJ) + (Post.pnu / Post.ptau) * np.dot(Post.w_J,Post.w_J)


    LGaussainReg = (
                    - 0.5 * LOGTWOPI \
                    + Post_cFunc - Prior_cFunc
                    - 0.5 * E_log_d * (Data.nDoc + Prior.pnu - Post.pnu)
                    - 0.5 * np.dot(E_d, (SS.yy + Prior.ptau + Prior.wPw_1 - Post.ptau - Post.wPw_1))
                    + np.dot(E_logdw, (SS.yEz_J + Prior.Pw_J - Post.Pw_J))
                    - 0.5 * np.sum(np.trace(np.dot(E_logdwTw, (SS.EzzT_JJ + Prior.P_JJ - Post.P_JJ))))

                    )
    return LGaussainReg


def c_Func(avec, K=0):
    ''' Evaluate cumulant function of the Dirichlet distribution

    Returns
    -------
    c : scalar real
    '''
    if isinstance(avec, float) or avec.ndim == 0:
        assert K > 0
        avec = avec * np.ones(K)
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    elif avec.ndim == 1:
        return gammaln(np.sum(avec)) - np.sum(gammaln(avec))
    else:
        return np.sum(gammaln(np.sum(avec, axis=1))) - np.sum(gammaln(avec))


#def calcPrior(Prior=None,pnu=12, ptau=1, w_J=None,
#        P_JJ=None, P_diag_J=None, P_diag_val=0.1, K=0,
#        **kwargs):
def calcPrior(Prior=None,pnu=12, ptau=1, w_J=None,
        P_JJ=None, P_diag_J=None, P_diag_val=0.1, K=0,
        **kwargs):
    ''' Initialize Prior attributes

    '''
    #print 'x Calc Prior'
    J = K
    # Init parameters of 1D Wishart prior on delta
    pnu = np.maximum(pnu, 1e-9)
    ptau = np.maximum(ptau, 1e-9)
    #print 'calc prior: pnu,ptau: ', pnu, ptau

    # Initialize precision matrix of the weight vector
    if P_JJ is not None:
        P_JJ = np.asarray(P_JJ)
    elif P_diag_J is not None:
        P_JJ = np.diag(np.asarray(P_diag_J))
    else:
        P_JJ = np.diag(P_diag_val * np.ones(J))
    assert P_JJ.ndim == 2
    assert P_JJ.shape == (J,J)

    # Initialize mean of the weight vector
    #print 'w_J', w_J
    if w_J is not None:
        w_J = np.asarray(w_J)
    else:
        #w_J = np.zeros(K)
        w_J = np.ones(K) #* 100
        #w_J = np.random.rand(K)
    #print 'w_J', w_J
    #assert w_J.ndim == 1
    #assert w_J.size == J
    Pw_J = np.dot(P_JJ, w_J)
    wPw_1 = np.dot(w_J, Pw_J)

    if Prior is None:
        Prior = ParamBag(K=K,J=K)
    Prior.setField('pnu', pnu, dims=None)
    Prior.setField('ptau', ptau, dims=None)
    Prior.setField('w_J', w_J, dims=('J'))
    Prior.setField('P_JJ', P_JJ, dims=('J', 'J'))
    Prior.setField('Pw_J', Pw_J, dims=('J'))
    Prior.setField('wPw_1', wPw_1, dims=None)
    #print Prior.w_J
    return Prior


def calcSummaryStats(Data, SS=None, LP=None, alpha=None,
                     doPrecompEntropy=0,
                     **kwargs):
    """ Calculate summary from local parameters for given data slice.

    Parameters
    -------
    Data : bnpy data object
    LP : local param dict with fields
            resp : Data.nObs x K array,
                    where resp[n,k] = posterior resp of comp k
            doPrecompEntropy : boolean flag
                    indicates whether to precompute ELBO terms in advance
                    used for memoized learning algorithms (moVB)

    Returns
    -------
    SS : SuffStatBag with K components
            * nDoc : scalar float
                    Counts total documents available in provided data.

            Also has optional ELBO field when precompELBO is True
            * Hvec : 1D array, size K
                    Vector of entropy contributions from each comp.
                    Hvec[k] = \sum_{n=1}^N H[q(z_n)], a function of 'resp'
    """
    #print 'x Calc Summary Stats'
    resp = LP['resp']
    _, K = resp.shape
    J = K
    Y_D = Data.response

    Ez_DJ = np.zeros((Data.nDoc,J)) #add bias term
    EzzT_DJJ = np.zeros((Data.nDoc,J,J))

    for d in xrange(Data.nDoc):
        start = int(Data.doc_range[d])
        stop = int(Data.doc_range[d+1])
        wc_d = Data.word_count[start:stop]
        N_d = int(sum(wc_d))

        resp_d = resp[start:stop,:]
        weighted_resp_d = wc_d[:,None] * resp_d

        Ez_DJ[d] = (1/float(N_d)) * np.sum(weighted_resp_d,axis=0)

        EzzT_DJJ[d] = calc_EzzT_one_doc_gaussianreg(wc_d,resp_d)


    S_yEz_J = np.sum(Y_D[:,np.newaxis] * Ez_DJ,axis=0)
    S_EzzT_JJ = np.sum(EzzT_DJJ,axis=0)
    S_yy = np.sum(np.square(Y_D))

    if SS is None:
        SS = SuffStatBag(K=K, J=K+1, D=Data.dim, nDoc=Data.nDoc)

    SS.setField('EzzT_JJ', S_EzzT_JJ, dims=('J','J'))
    SS.setField('yEz_J', S_yEz_J, dims=('J'))
    SS.yy = S_yy
    '''
    SS = SuffStatBag(K=K, D=Dslice.dim)
    SS.setField('nDoc', Dslice.nDoc, dims=None)
    if doPrecompEntropy:
            Hvec = L_entropy(Dslice, LP, returnVector=1)
            Lalloc = L_alloc(Dslice, LP, alpha=alpha)
            Lslda = L_sLDA(Dslice, LP)
            SS.setELBOTerm('Hvec', Hvec, dims='K')
            SS.setELBOTerm('L_alloc', Lalloc, dims=None)
            SS.setELBOTerm('Lslda', Lslda)
    '''
    return SS


def update_global_params(SS,Prior,Post=None):
    #print 'x Update Global Params'
    pnu_post = SS.nDoc + Prior.pnu

    P_JJ_post = Prior.P_JJ + SS.EzzT_JJ

    #print SS.yEz_J
    #print P_JJ_post
    #print np.linalg.inv(P_JJ_post)
    w_J_post = np.dot(np.linalg.inv(P_JJ_post), (Prior.Pw_J + SS.yEz_J))

    wPw_1_post = np.dot(np.dot(w_J_post,P_JJ_post),w_J_post)
    ptau_post = Prior.ptau + SS.yy + Prior.wPw_1 - wPw_1_post

    Pw_J_Post = np.dot(P_JJ_post, w_J_post)

    if Post is None:
        Post = ParamBag(K=Prior.K, D=Prior.D)

    K = w_J_post.size
    J = K

    Post.setField('J', J, dims=None)
    Post.setField('K', K, dims=None)
    Post.setField('pnu', pnu_post, dims=None)
    Post.setField('ptau', ptau_post, dims=None)

    Post.setField('w_J', w_J_post, dims='J')
    Post.setField('P_JJ', P_JJ_post, dims=('J', 'J'))
    Post.setField('Pw_J', Pw_J_Post, dims=('J'))
    Post.setField('wPw_1', wPw_1_post, dims=None)
    Post.setField('mean_weights', w_J_post, dims='J')
    #Post.setField('delta', ptau_post / (pnu_post - 2), dims=None)
    Post.setField('stdev', ptau_post / (pnu_post - 2), dims=None)
    return Post


def calc_EzzT_one_doc_gaussianreg(wc_d,resp_d):

    nTokens_d, K = resp_d.shape
    weighted_resp_d = wc_d[:,None] * resp_d #Doc
    N_d = np.sum(wc_d)

    A = np.ones((nTokens_d,nTokens_d)) - np.eye(nTokens_d)

    EZTZ = np.inner(weighted_resp_d.transpose(),A)
    EZTZ = np.inner(EZTZ,weighted_resp_d.transpose())

    EZTZ = EZTZ + np.diag(np.sum(wc_d[:,None] * weighted_resp_d,axis=0))
    EZTZ = (1.0/np.square(N_d)) * EZTZ
    return EZTZ


def E_mahal_dist_D(Y_D, Ez_DJ, EzzT_DJJ,
                pnu=None,
                ptau=None,
                w_J=None,
                P_JJ=None):
    ''' Calculate expected mahalanobis distance under regression model

    For each data index n, computes expected distance using provided
    cluster-specific parameters:
    $$
    d_n = E[ delta_k (y_n - w_k^T x_n)^2 ]
    $$

    Returns
    -------
    d_N : 1D array, size N
    '''
    E_log_d = digamma(0.5 * pnu) - np.log(0.5 * ptau)
    E_d = (pnu / ptau)




    E_logdw = E_log_d* w_J
    E_logdwTw = np.linalg.inv(P_JJ) + (pnu / ptau) * np.dot(w_J,w_J)


    E_mahal_dist_D = E_d * Y_D**2
    E_mahal_dist_D -= 2*Y_D*Ez_DJ*Elogdw
    E_mahal_dist_D += EzzT_DJJ * E_logdwTw

    return E_mahal_dist_D.sum()

def cFunc_logreg(pnu=1e-9, ptau=1e-9, w_J=None, P_JJ=None, logdet_P_JJ=None):
    ''' Compute cumulant function for Multivariate-Normal-Univariate-Wishart

    Returns
    -------
    c : float
        scalar output of cumulant function
    '''
    if logdet_P_JJ is None:
        logdet_P_JJ = np.log(np.linalg.det(P_JJ))
    J = w_J.size
    c_wish_1dim = (0.5 * pnu) * np.log(0.5 * ptau) - gammaln(0.5 * pnu)
    c_normal_Jdim = - 0.5 * J * LOGTWOPI + 0.5 * logdet_P_JJ
    return c_wish_1dim + c_normal_Jdim
