'''
SMSB.py

Finite single membership stochastic block model.  Follows generative process:

For all nodes i:
   Draw pi_i ~ Dirichlet(gamma)
   Draw z_i ~ Categorical(pi_i)
   For all other nodes j:
      Draw x_{ij} ~ Bernouli(w_{z_i, z_j})

For l,m = 1,...,K:
   w_{lm} ~ Beta(tau_1, tau_0)
'''

from builtins import *
import numpy as np
from bnpy.util import logsumexp
from bnpy.util import gammaln, digamma, EPS
from bnpy.suffstats import SuffStatBag
from bnpy.util import StateSeqUtil

from bnpy.allocmodel.mix.FiniteMixtureModel import FiniteMixtureModel
from bnpy.allocmodel import AllocModel


class FiniteSMSB(FiniteMixtureModel):

    ''' Single membership stochastic block model, with K components.

    Attributes
    -------
    * inferType : string {'EM', 'VB', 'moVB', 'soVB'}
        indicates which updates to perform for local/global steps
    * K : int
        number of components
    * alpha : float
        scalar symmetric Dirichlet prior on mixture weights

    Attributes for VB
    ---------
    TODO
    '''

    def __init__(self, inferType, priorDict=dict()):
        self.EStepLaps = 25
        super(FiniteSMSB, self).__init__(inferType, priorDict)
        self.set_prior(**priorDict)
        self.K = 0
        self.Npair = 0
        self.hamming = 0
        self.inferType = inferType

        self.estZ = 0

    def getSSDims(self):
        ''' Get dimensions of interactions between components.

        Overrides default of ('K',), as we need E_log_soft_ev to be
           dimension E x K x K
        '''
        return ('K', 'K',)

    def calc_local_params(self, Data, LP, **kwargs):
        ''' Calculate local parameters for each data item and each component.

        This is part of the E-step.
        Note that this is the main place we differ from FiniteMixtureModel.py

        Args
        -------
        Data : bnpy data object with Data.nObs observations
        LP : local param dict with fields
              E_log_soft_ev : Data.nObs x K x K array
                  E_log_soft_ev[n,l,m] = log p(data obs n | comps l, m)

        Returns
        -------
        LP : local param dict with fields
             resp : 2D array, size Data.nObs x K array
                    resp[n,l,m] = posterior responsibility comps. l,m have for
                    item n
        '''

        if self.inferType.count('EM') > 0:
            raise NotImplementedError(
                'EM not implemented for FiniteSMSB (yet)')

        N = Data.nNodes
        K = self.K
        logSoftEv = LP['E_log_soft_ev']  # E x K x K
        logSoftEv[np.where(Data.sourceID == Data.destID), :, :] = 0
        logSoftEv = np.reshape(logSoftEv, (N, N, K, K))

        if 'respSingle' not in LP:
            LP['respSingle'] = np.ones((N, K)) / K
        resp = LP['respSingle']

        Elogpi = digamma(self.theta) - digamma(np.sum(self.theta))  # Size K

        respTerm = np.zeros(K)
        for lap in range(self.EStepLaps):
            for i in range(Data.nNodes):
                respTerm = np.einsum(
                    'jlm,jm->l', logSoftEv[i, :, :, :], resp) + \
                    np.einsum('jlm,jl->m', logSoftEv[:, i, :, :], resp)
                resp[i, :] = np.exp(Elogpi + respTerm)
                resp[i, :] /= np.sum(resp[i, :])

        # For now, do the stupid thing of building the N^2 x K resp matrix
        #   (soon to change when using sparse data)
        # np.einsum makes fullResp[i,j,l,m] = resp[i,l]*resp[j,m]
        fullResp = np.einsum('il,jm->ijlm', resp, resp)
        fullResp = fullResp.reshape((N**2, K, K))
        fullResp[np.where(Data.sourceID == Data.destID), :, :] = 0
        LP['resp'] = fullResp
        LP['respSingle'] = resp
        self.make_hard_asgn_local_params(Data, LP)

        return LP

    def make_hard_asgn_local_params(self, Data, LP):
        ''' Convert soft assignments to hard assignments.

        Returns
        --------
        LP : local params dict, with new fields
             Z : 1D array, size N
                    Z[n] is an integer in range {0, 1, 2, ... K-1}
             resp : 2D array, size N x K+1 (with final column empty)
                    resp[n,k] = 1 iff Z[n] == k
        '''
        Z = np.argmax(LP['respSingle'], axis=1)
        self.estZ = Z

    def get_global_suff_stats(self, Data, LP, doPrecompEntropy=None, **kwargs):
        ''' Calculate the sufficient statistics for global parameter updates

        Only adds stats relevant for this allocModel.
        Other stats are added by the obsModel.

        Args
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
        SS : SuffStats for K components, with field
              N : vector of dimension K,
                   effective number of observations assigned to each comp
              Npair : matrix of dimensions K x K, where Npair[l,m] =
                      effective # of obs x_{ij} with z_{il} and z_{jm}

        '''
        Npair = np.sum(LP['resp'], axis=0)
        self.Npair = Npair
        N = np.sum(LP['respSingle'], axis=0)

        SS = SuffStatBag(K=N.shape[0], D=Data.dim)
        SS.setField('Npair', Npair, dims=('K', 'K'))
        SS.setField('N', N, dims=('K',))
        if doPrecompEntropy is not None:
            ElogqZ_vec = self.E_logqZ(LP)
            SS.setELBOTerm('ElogqZ', ElogqZ_vec, dims=('K',))
        return SS

    def E_logqZ(self, LP):
        return np.sum(
            LP['respSingle'] * np.log(LP['respSingle'] + EPS), axis=0)

    def to_dict(self):
        myDict = super(FiniteSMSB, self).to_dict()

        myDict['Npair'] = self.Npair
        myDict['estZ'] = self.estZ
        return myDict
