'''
FiniteAssortativeMMSB.py

Assortative mixed membership stochastic blockmodel.
'''
from builtins import *
import numpy as np
import itertools

from bnpy.allocmodel import AllocModel
from bnpy.suffstats import SuffStatBag
from bnpy.util import gammaln, digamma, EPS

from bnpy.util import StickBreakUtil
from bnpy.allocmodel.topics import OptimizerRhoOmegaBetter
from bnpy.allocmodel.topics.HDPTopicUtil import c_Beta, c_Dir, L_top

from .FiniteAssortativeMMSB import FiniteAssortativeMMSB
from .HDPMMSB import updateRhoOmega, updateThetaAndThetaRem, _beta2rhoomega
from .HDPMMSB import initRhoOmegaFromScratch, initThetaFromScratch

class HDPAssortativeMMSB(FiniteAssortativeMMSB):

    """ Assortative version of MMSB, with HDP prior.

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
    * theta : 1D array, size K
        Estimated parameters for Dirichlet posterior over mix weights
        theta[k] > 0 for all k
    """

    def __init__(self, inferType, priorDict=dict()):
        super(HDPAssortativeMMSB, self).__init__(inferType, priorDict)

    def set_prior(self, alpha=0.5, gamma=10, epsilon=0.05):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def getCompDims(self):
        ''' Get dimensions of latent component interactions.

        Assortative models use only K states.

        Returns
        -------
        dims : tuple
        '''
        return ('K',)

    def E_logPi(self, returnRem=0):
        ''' Compute expected probability \pi for each node and state

        Returns
        -------
        ElogPi : nNodes x K
        '''
        digammasumtheta = digamma(
            self.theta.sum(axis=1) + self.thetaRem)
        ElogPi = digamma(self.theta) - digammasumtheta[:, np.newaxis]
        if returnRem:
            ElogPiRem = digamma(self.thetaRem) - digammasumtheta
            return ElogPi, ElogPiRem
        return ElogPi

    # calc_local_params inherited from FiniteAssortativeMMSB
    # get_global_suff_stats inherited from FiniteAssortativeMMSB

    def update_global_params_VB(self, SS, **kwargs):
        ''' Update global parameter theta to optimize VB objective.

        Post condition
        --------------
        Attributes rho,omega,theta set to optimal value given suff stats.
        '''
        nGlobalIters = 2
        nNode = SS.NodeStateCount.shape[0]

        if not hasattr(self, 'rho') or self.rho.size != SS.K:
            self.rho = OptimizerRhoOmegaBetter.make_initrho(
                SS.K, nNode, self.gamma)
        self.omega = OptimizerRhoOmegaBetter.make_initomega(
            SS.K, nNode, self.gamma)
        # Update theta first, so it reflects most recent NodeStateCounts
        self.theta, self.thetaRem = updateThetaAndThetaRem(
            SS, rho=self.rho, alpha=self.alpha, gamma=self.gamma)
        # Now, alternatively update rho and theta...
        for giter in range(nGlobalIters):
            self.rho, self.omega = updateRhoOmega(
                theta=self.theta, thetaRem=self.thetaRem,
                initrho=self.rho, omega=self.omega,
                alpha=self.alpha, gamma=self.gamma)
            self.theta, self.thetaRem = updateThetaAndThetaRem(
                SS, rho=self.rho, alpha=self.alpha, gamma=self.gamma)

    def set_global_params(self, hmodel=None,
                          rho=None, omega=None, theta=None, thetaRem=None,
                          **kwargs):
        ''' Set rho, omega, theta to specific provided values.
        '''
        if hmodel is not None:
            self.K = hmodel.allocModel.K
            if hasattr(hmodel.allocModel, 'rho'):
                self.rho = hmodel.allocModel.rho
                self.omega = hmodel.allocModel.omega
            else:
                raise AttributeError('Unrecognized hmodel. No field rho.')
            if hasattr(hmodel.allocModel, 'theta'):
                self.theta = hmodel.allocModel.theta
                self.thetaRem = hmodel.allocModel.thetaRem
            else:
                raise AttributeError('Unrecognized hmodel. No field theta.')
        elif rho is not None \
                and omega is not None \
                and theta is not None:
            self.rho = rho
            self.omega = omega
            self.theta = theta
            self.thetaRem = thetaRem
        else:
            self.rho, self.omega = initRhoOmegaFromScratch(**kwargs)
            self.theta, self.thetaRem = initThetaFromScratch(rho=rho, **kwargs)
        self.K = self.rho.size
        assert self.K == self.omega.size
        assert self.K == self.theta.shape[-1]

    def init_global_params(self, Data, K=0, initLP=None, **kwargs):
        ''' Initialize global parameters "from scratch" to reasonable values.

        Post condition
        --------------
        Global parameters rho, omega, theta, thetaRem set to
        valid values.
        '''
        self.K = K
        initbeta = (1.0 - 0.01)/K * np.ones(K)
        assert np.sum(initbeta) < 1.0
        self.rho, self.omega = _beta2rhoomega(
            beta=initbeta, K=K,
            nDoc=Data.nNodes, gamma=self.gamma)

        if initLP is not None:
            # Create optimal theta for provided initial local params
            initSS = self.get_global_suff_stats(Data, initLP)
            self.theta, self.thetaRem = updateThetaAndThetaRem(
                K=K, NodeStateCount=initSS.NodeStateCount,
                rho=self.rho, alpha=self.alpha, gamma=self.gamma)
        else:
            # Create theta from scratch
            self.theta, self.thetaRem = initThetaFromScratch(
                Data=Data, rho=rho, alpha=self.alpha, gamma=self.gamma)

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        ''' Compute training objective function on provided input.

        Returns
        -------
        L : scalar float
        '''
        Lalloc = self.L_alloc_no_slack()
        Lslack = self.L_slack(SS)
        # Compute entropy term
        if SS.hasELBOTerm('Hresp'):
            Lentropy = SS.getELBOTerm('Hresp').sum() + \
                SS.getELBOTerm('Hresp_bg')
        else:
            Lentropy = self.L_entropy(LP)

        if SS.hasELBOTerm('Ldata_bg'):
            Lbgdata = SS.getELBOTerm('Ldata_bg')
        else:
            Lbgdata = LP['Ldata_bg']
        if todict:
            return dict(Lentropy=Lentropy,
                Lalloc=Lalloc, Lslack=Lslack,
                Lbgdata=Lbgdata)
        return Lalloc + Lentropy + Lslack + Lbgdata


    def L_alloc_no_slack(self):
        ''' Compute allocation term of objective function, without slack term

        Returns
        -------
        L : scalar float
        '''
        prior_cDir = L_top(nDoc=self.theta.shape[0],
            alpha=self.alpha, gamma=self.gamma,
            rho=self.rho, omega=self.omega)
        post_cDir = c_Dir(self.theta, self.thetaRem)
        return prior_cDir - post_cDir

    def L_slack(self, SS):
        ''' Compute slack term of the allocation objective function.

        Returns
        -------
        L : scalar float
        '''
        ElogPi, ElogPiRem = self.E_logPi(returnRem=1)
        Ebeta = StickBreakUtil.rho2beta(self.rho, returnSize='K')
        Q = SS.NodeStateCount + self.alpha * Ebeta - self.theta
        Lslack = np.sum(Q * ElogPi)

        alphaEbetaRem = self.alpha * (1.0 - Ebeta.sum())
        LslackRem = np.sum((alphaEbetaRem - self.thetaRem) * ElogPiRem)
        return Lslack + LslackRem

    def to_dict(self):
        return dict(theta=self.theta, thetaRem=self.thetaRem,
            rho=self.rho, omega=self.omega)

    def from_dict(self, myDict):
        self.inferType = myDict['inferType']
        self.K = myDict['K']
        self.rho = myDict['rho']
        self.omega = myDict['omega']
        self.theta = myDict['theta']
        self.thetaRem = myDict['thetaRem']

    def get_prior_dict(self):
        return dict(alpha=self.alpha, gamma=self.gamma, epsilon=self.epsilon)


def calcSummaryStatsForMerge(Data, LP, mUIDPairs):
    ''' Compute stats that represent a merge.
    '''
    pass
