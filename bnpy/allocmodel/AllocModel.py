''' AllocModel.py
'''
from __future__ import division


class AllocModel(object):

    def __init__(self, inferType):
        self.inferType = inferType

    def set_prior(self, **kwargs):
        pass

    def get_keys_for_memoized_local_params(self):
        ''' Return LP field names required for warm starts of local step
        '''
        return list()

    
    def getCompDims(self):
        ''' Get the dimensions of the latent clusters for this object.

        Returns
        -------
        dimTuple : tuple with dimensions of latent clusters.
        '''
        return ('K',)

    def calc_local_params(self, Data, LP):
        ''' Compute local parameters for each data item and component.

        This is the E-step of EM algorithm.

        Returned LP contains optimal values of local parameters
        specific to the provided dataset. 
        Updated values computed using current global parameter attributes.

        Possible keyword arguments control model-specific computations.

        Args
        ----
        Data : :class:`.DataObj`
            Dataset to compute local parameters for.
        LP : dict
            Must contain cond. likelihoods in field 'E_log_soft_ev',
            a 2D array that is N x K provided by the observation model.

        Returns
        -------
        LP : dict
            Contains updated fields for all K clusters in current model.
            * 'resp' : N x K 2D array, soft assignments for each data atom.
        '''
        pass

    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
        ''' Compute low-dim summaries for provided local params.

        Returned sufficient statistics are deterministic given Data, LP.

        Possible keyword arguments control model-specific computations.

        Args
        ----
        Data : :class:`.DataObj`
            Dataset to be summarized.
        SS : :class:`.SuffStatBag`
            If present, all summaries will be added to this bag.
            If None, new bag will be created and returned.
        LP : dict
            Holds valid local params for K' clusters and all atoms in Data.

        Returns
        -------
        SS : :class:`.SuffStatBag`
            Updated fields for each of K' clusters represented in LP
        '''
        pass

    def update_global_params(self, SS, rho=None, **kwargs):
        ''' Update global parameter attributes for this model.

        This is the M-step of EM algorithm.

        Args
        ----
        SS : :class:`.SuffStatBag`
            Sufficient statistics needed for update.

        Returns
        -------
        None

        Post Condition
        --------------
        Attribute K reset to the number of active clusters in SS.
        Global parameter attributes updated in-place or reallocated.
        '''
        self.K = SS.K
        if self.inferType == 'EM':
            self.update_global_params_EM(SS)
        elif self.inferType == 'VB' or self.inferType.count('moVB'):
            self.update_global_params_VB(SS, **kwargs)
        elif self.inferType == 'GS':
            self.update_global_params_VB(SS, **kwargs)
        elif self.inferType == 'soVB':
            if rho is None or rho == 1:
                self.update_global_params_VB(SS, **kwargs)
            else:
                self.update_global_params_soVB(SS, rho, **kwargs)
        else:
            raise ValueError(
                'Unrecognized Inference Type! %s' % (self.inferType))

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        """ Calculate ELBO objective function value for provided state.

        Args
        ----
        Data : optional,
            If not provided, relies exclusively on summaries in SS
        SS : :class:`.SuffStatBag`
            Contains valid summaries for desired dataset.
        LP : optional, dict
            If not provided, relies exclusively on summaries in SS
            If provided, used in place of summaries in SS when possible.

        Keyword Args
        ------------
        todict : boolean
            If True, return a dict with different ELBO terms
                under named keys like 'Ldata' and 'Lentropy'
            If False [default], return scalar value equal to sum of terms.

        Returns
        -------
        L : float
            Represents sum of all terms in optimization objective.
            Will be a dict if todict option is True.
        """
        pass

    def calcELBOFromLP(self, Data, LP):
        """ Calculate ELBO value for provided data & local parameters

        TODO implement this
        """
        pass

    def calcELBOFromSS(self, SS):
        """ Calculate ELBO value for provided sufficient stats.

        TODO implement this
        """
        pass

    def get_info_string(self):
        ''' Returns one-line human-readable terse description of this object
        '''
        pass

    def sample_local_params(self, obsModel, Data, SS, LP):
        ''' Sample local assignments for each data item.
        '''
        pass

    def to_dict_essential(self):
        PDict = dict(name=self.__class__.__name__, inferType=self.inferType)
        if hasattr(self, 'K'):
            PDict['K'] = self.K
        return PDict

    def to_dict(self):
        pass

    def from_dict(self):
        pass

    def get_prior_dict(self):
        pass

    def make_hard_asgn_local_params(self, LP):
        ''' Convert soft to hard assignments for provided local params

        Parameters
        --------
        LP : dict
            Local parameters as key/value string/array pairs
            * resp : 2D array, size N x K
        '''
        LP['Z'] = np.argmax(LP['resp'], axis=1)
        K = LP['resp'].shape[1]
        LP['resp'].fill(0)
        for k in xrange(K):
            LP['resp'][LP['Z'] == k, k] = 1
        return LP

    def getHandleCalcLocalParams(self):
        return self.calc_local_params

    def getHandleCalcSummaryStats(self):
        return self.get_global_suff_stats
