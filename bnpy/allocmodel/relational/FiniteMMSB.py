'''
FiniteMMSB.py
'''
from builtins import *
import numpy as np

from scipy.sparse import csc_matrix
from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS
from bnpy.util.NumericUtil import calcRlogR
from bnpy.allocmodel.topics.HDPTopicUtil import c_Dir


class FiniteMMSB(AllocModel):

    """ Mixed membership stochastic block model, with K components.

    Attributes
    -------
    inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    K : int
        number of components
    alpha : float
        scalar symmetric Dirichlet prior on mixture weights
        pi_v ~ Dir( alpha/K, alpha/K, ... alpha/K)

    Attributes for VB
    ---------
    theta : 2D array, nNodes x K
        theta[n,:] gives parameters for Dirichlet variational factor
        defining distribution over membership probabilities for node n
    """

    def __init__(self, inferType, priorDict=dict()):
        if inferType.count('EM') > 0:
            raise NotImplementedError(
                'EM not implemented for FiniteMMSB (yet)')

        self.inferType = inferType
        self.set_prior(**priorDict)
        self.K = 0

        # Variational parameter for pi
        self.theta = None

    def set_prior(self, alpha=.1):
        self.alpha = float(alpha)

    def get_active_comp_probs(self):
        print('TODO')

    def getCompDims(self):
        ''' Get dimensions of latent component interactions.

        Overrides default of ('K',), since E_log_soft_ev needs to be ('K','K')

        Returns
        -------
        dims : tuple
        '''
        return ('K', 'K',)

    def E_logPi(self):
        ''' Compute expected value of log \pi for each node and state.

        Returns
        -------
        ElogPi : 2D array, nNodes x K
        '''
        sumtheta = self.theta.sum(axis=1)
        ElogPi = digamma(self.theta) - digamma(sumtheta)[:, np.newaxis]
        return ElogPi

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Compute local parameters for provided dataset.

        Args
        -------
        Data : GraphData object
        LP : dict
            Local parameters, must contain fields
            * E_log_soft_ev : nEdges x K x K

        Returns
        -------
        LP : dict
            Local parameters, will contain updated fields:
            * resp : nEdges x K x K
                resp[e,j,k] = prob that edge e is explained by
                connection from state/block j to block k
        '''
        if self.inferType.count('EM') > 0:
            raise NotImplementedError("TODO")
        if Data.isSparse:  # Sparse binary data.
            raise NotImplementedError("TODO")

        K = self.K
        ElogPi = self.E_logPi()

        # resp : nEdges x K x K
        #    resp[e(s,t),k,l] = ElogPi[s,k] + ElogPi[t,l] + likelihood
        resp = ElogPi[Data.edges[:,0], :, np.newaxis] + \
               ElogPi[Data.edges[:,1], np.newaxis, :]
        logSoftEv = LP['E_log_soft_ev']  # E x K x K
        resp += logSoftEv

        # In-place exp and normalize
        resp -= np.max(resp, axis=(1,2))[:, np.newaxis, np.newaxis]
        np.exp(resp, out=resp)
        resp /= resp.sum(axis=(1,2))[:, np.newaxis, np.newaxis]
        np.maximum(resp, 1e-100, out=resp)
        LP['resp'] = resp
        return LP

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=0, **kwargs):
        ''' Compute sufficient stats for provided dataset and local params

        Returns
        -------
        SS : SuffStatBag
            Updated fields
            * NodeStateCount : 2D array, nNodes x K
            * N : 2D array, size K x K
        '''
        K = LP['resp'].shape[-1]

        V = Data.nNodes
        SS = SuffStatBag(K=K, D=Data.dim, V=V)

        # NodeStateCount_src[i,k]
        #   Num edges assigned to topic k associated with node i as source
        srcResp = LP['resp'].sum(axis=2)
        NodeStateCount_src = Data.getSparseSrcNodeMat() * srcResp
        # Equivalent but slower: for loop
        # NodeStateCount_src = np.zeros((Data.nNodes, K))
        # for i in xrange(Data.nNodes):
        #     mask_i = Data.edges[:,0] == i
        #     NodeStateCount_src[i,:] = srcResp[mask_i].sum(axis=0)

        # NodeStateCount_rcv[i,k]
        #   Num edges assigned to topic k associated with node i as receiver
        rcvResp = LP['resp'].sum(axis=1)
        NodeStateCount_rcv = Data.getSparseRcvNodeMat() * rcvResp

        # Summing src counts and rcv counts gives the total
        SS.setField(
            'NodeStateCount', NodeStateCount_src + NodeStateCount_rcv,
            dims=('V', 'K'))
        # Compute total atoms assigned to each cluster pair
        Nresp = np.sum(LP['resp'], axis=0)
        SS.setField('N', Nresp, dims=('K','K'))

        if doPrecompEntropy:
            # Remember, resp has shape nEdges x K x K
            # So, need to sum so we track scalar entropy, not K x K
            Hresp = calcLentropyAsScalar(LP)
            SS.setELBOTerm('Hresp', Hresp, dims=None)
        return SS

    def forceSSInBounds(self, SS):
        ''' Force certain fields in bounds, to avoid numerical issues.

        Returns
        -------
        Nothing.  SS is updated in-place.
        '''
        np.maximum(SS.NodeStateCount, 0, out=SS.NodeStateCount)

    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameter theta to optimize VB objective.

        Post condition
        --------------
        Attribute theta set to optimal value given suff stats.
        '''
        self.theta = self.alpha / SS.K + SS.NodeStateCount

    def set_global_params(self, hmodel=None, theta=None, **kwargs):
        ''' Set global parameters to specific values.

        Post condition
        --------------
        Attributes theta, K set to provided values.
        '''
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = hmodel.allocModel.theta
        else:
            if self.inferType == 'EM':
                raise NotImplemetedError(
                    'EM not implemented (yet) for FiniteMMSB')
            elif self.inferType.count('VB') > 0:
                self.theta = theta
                self.K = theta.shape[-1]

    def init_global_params(self, Data, K=0, initLP=None, **kwargs):
        ''' Initialize global parameters "from scratch" to reasonable values.

        Post condition
        --------------
        Attributes theta, K set to reasonable values.
        '''
        self.K = K
        if initLP is not None:
            # Compute NodeStateCount from provided initial local params
            initSS = self.get_global_suff_stats(Data, initLP)
            self.theta = self.alpha / K + initSS.NodeStateCount
        else:
            # Create random initNodeStateCount values
            # by drawing from Dirichlet prior on pi_v, scaled by nEdgesPerNode
            PRNG = np.random.RandomState(K)
            piMean = self.alpha / K * np.ones(K)
            nEdgesPerNode = Data.getSparseSrcNodeMat().sum(axis=1) + \
                Data.getSparseRcvNodeMat().sum(axis=1)
            initNodeStateCount = nEdgesPerNode * \
                PRNG.dirichlet(piMean, size=Data.nNodes)
            self.theta = self.alpha / K + initNodeStateCount

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        ''' Compute training objective function on provided input.

        Returns
        -------
        L : scalar float
        '''
        Lalloc = self.L_alloc_no_slack()
        Lslack = self.L_slack(SS)
        if SS.hasELBOTerm('Hresp'):
            Lentropy = SS.getELBOTerm('Hresp')
        else:
            Lentropy = calcLentropyAsScalar(LP)
        if todict:
            return dict(Lentropy=Lentropy, Lalloc=Lalloc, Lslack=Lslack)
        return Lalloc + Lentropy + Lslack

    def L_alloc_no_slack(self):
        ''' Compute allocation term of objective function, without slack term

        Returns
        -------
        L : scalar float
        '''
        N = self.theta.shape[0]
        K = self.K
        prior_cDir = N * (gammaln(self.alpha) - K * gammaln(self.alpha/K))
        post_cDir = np.sum(gammaln(np.sum(self.theta, axis=1))) - \
            np.sum(gammaln(self.theta))
        return prior_cDir - post_cDir

    def L_slack(self, SS):
        ''' Compute slack term of the allocation objective function.

        Returns
        -------
        L : scalar float
        '''
        ElogPi = digamma(self.theta) - \
            digamma(np.sum(self.theta, axis=1))[:, np.newaxis]
        Q = SS.NodeStateCount + self.alpha / SS.K - self.theta
        Lslack = np.sum(Q * ElogPi)
        return Lslack

    def L_entropy_as_scalar(self, LP):
        ''' Compute entropy term of objective as a scalar.

        Returns
        -------
        Hresp : scalar
        '''
        return calcLentropyAsScalar(LP)

    def to_dict(self):
        return dict(theta=self.theta)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.theta = myDict['theta']

    def get_prior_dict(self):
        return dict(alpha=self.alpha)


    def calc_estZ(self):
        ''' Calculate hard assignment for each node.

        Returns
        -------
        Z : 1D array, size nNodes
            indicator for which cluster each node most belongs to
        '''
        return np.argmax(self.theta, axis=1)


def calcLentropyAsScalar(LP):
    ''' Compute entropy term of objective as a scalar.

    Returns
    -------
    Hresp : scalar
    '''
    return -1.0 * np.sum(calcRlogR(LP['resp']))

    '''
    def initLPFromTruth(self, Data):
        K = np.max(Data.TrueParams['Z']) + 1
        N = Data.nNodes
        Z = Data.TrueParams['Z']
        resp = np.zeros((N, N, K, K))
        for i in xrange(N):
            for j in xrange(N):
                resp[i, j, Z[i, j, 0], Z[j, i, 0]] = 1
        diag = np.diag_indices(N)
        resp[diag[0], diag[1], :, :] = 0
        squareResp = resp
        resp = np.reshape(resp, (N**2, K, K))
        LP = {'resp': resp, 'squareResp': squareResp}

        if Data.isSparse:
            LP['Count1'] = np.sum(squareResp[Data.respInds[:, 0],
                                             Data.respInds[:, 1]], axis=0)
            LP['Count0'] = np.sum(squareResp, axis=(0, 1)) - LP['Count1']

        return LP
    '''
