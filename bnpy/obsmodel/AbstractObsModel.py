import copy

class AbstractObsModel(object):

    ''' Generic parent class for observation/data-generation models.

    Implements basic functionality common to all models, such as
    * determining subroutines to call based on inferType (EM or VB)
    * updating global parameters (again depending on EM or VB)
    * caching of temporary "helper" functions, for fast re-use
    '''

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
        if LP is None:
            LP = dict()
        LP['obsModelName'] = str(self.__class__.__name__)
        if self.inferType == 'EM':
            LP['E_log_soft_ev'] = self.calcLogSoftEvMatrix_FromEstParams(
                Data, **kwargs)
        else:
            L = self.calcLogSoftEvMatrix_FromPost(
                Data, **kwargs)
            if isinstance(L, dict):
                LP.update(L)
            else:
                LP['E_log_soft_ev'] = L
        return LP

    def get_global_suff_stats(self, Data, SS, LP, **kwargs):
        """ Compute sufficient statistics for provided local parameters.

        Returns
        ----
        SS : bnpy.suffstats.SuffStatBag
            Updated in place from provided value of SS.
        """
        SS = self.calcSummaryStats(Data, SS, LP, **kwargs)
        return SS

    def update_global_params(self, SS, rho=None, **kwargs):
        """ Update parameters to maximize objective given suff stats.

        Post Condition
        -------
        Either EstParams or Post attributes updated in place.
        """
        if self.inferType == 'EM':
            return self.updateEstParams_MaxLik(SS)
        elif rho is not None and rho < 1.0:
            return self.updatePost_stochastic(SS, rho)
        else:
            return self.updatePost(SS)

    def set_global_params(self, **kwargs):
        ''' Set global parameters to specific provided values.

        This method provides overall governing logic for setting
        the global parameter attributes of this model.

        If we are doing point-estimate (EM) learning, then
        these values fill the ParamBag attribute 'EstParams'.

        Otherwise, if we are doing variational learning, then
        these values fill the ParamBag attribute 'Post'.

        Post Condition
        ---------
        Exactly one of Post or EstParams will be updated in-place.
        '''
        if 'hmodel' in kwargs:
            hmodel = kwargs['hmodel']
            if hasattr(hmodel.obsModel, "Post"):
                self.setPostFactors(obsModel=hmodel.obsModel)
                return
            elif hasattr(hmodel.obsModel, "EstParams"):
                self.setEstParams(obsModel=hmodel.obsModel)
                return

        # First, try setEstParams, and fall back on setPost on any trouble
        didSetPost = 0
        try:
            self.setEstParams(**kwargs)
        except:
            try:
                self.setPostFactors(**kwargs)
                didSetPost = 1
            except:
                raise ValueError('Unrecognised args for set_global_params')

        # Make sure EM methods have an EstParams field
        if self.inferType == 'EM' and didSetPost:
            self.setEstParamsFromPost(self.Post, **kwargs)
            del self.Post

        # Make sure VB methods have a Post field
        if self.inferType != 'EM' and not didSetPost:
            self.setPostFromEstParams(self.EstParams, **kwargs)
            del self.EstParams

    def calc_evidence(self, Data, SS, LP, todict=0, **kwargs):
        """ Evaluate objective function at provided state.

        Returns
        -----
        L : float
        """
        if self.inferType == 'EM':
            # Handled entirely by evidence field of LP dict
            # which  is used in the allocation model.
            return 0
        else:
            if todict:
                return dict(Ldata=self.calcELBO_Memoized(SS, **kwargs))
            return self.calcELBO_Memoized(SS, **kwargs)

    def to_dict(self):
        """ Convert all attributes to dictionary for pickling/storage.

        Returns
        -----
        d : dict
        """
        PDict = dict(name=self.__class__.__name__,
                     inferType=self.inferType)
        if hasattr(self, 'EstParams'):
            PDict['K'] = self.EstParams.K
            PDict['D'] = self.EstParams.D
            for key in self.EstParams._FieldDims.keys():
                PDict[key] = getattr(self.EstParams, key)
        if hasattr(self, 'Post'):
            PDict['K'] = self.Post.K
            PDict['D'] = self.Post.D
            for key in self.Post._FieldDims.keys():
                PDict[key] = getattr(self.Post, key)
        return PDict

    def get_prior_dict(self):
        """ Convert all prior hyperparameters to dict for pickling/storage.

        Returns
        -----
        d : dict
        """
        PDict = dict()
        PDict['name'] = self.__class__.__name__
        if hasattr(self, 'min_covar'):
            PDict['min_covar'] = self.min_covar
        if hasattr(self, 'inferType'):
            PDict['inferType'] = self.inferType
        if hasattr(self, 'CompDims'):
            PDict['CompDims'] = self.CompDims

        if hasattr(self, 'Prior'):
            all_keys = self.Prior.__dict__.keys()
            field_names = self.Prior._FieldDims.keys()
            other_names = [key for key in all_keys if key not in field_names]
            for key in field_names:
                PDict[key] = getattr(self.Prior, key)
            for key in other_names:
                PDict[key] = getattr(self.Prior, key)
        return PDict

    def GetCached(self, key, k=None):
        ''' Evaluate function provided, using cached value if possible.

        Allows smart reuse of expectation calculations.
        '''
        ckey = key + '-' + str(k)
        try:
            return self.Cache[ckey]
        except KeyError:
            Val = getattr(self, '_' + key)(k)
            self.Cache[ckey] = Val
            return Val

    def ClearCache(self):
        ''' Remove all values from the function cache.
        '''
        self.Cache.clear()

    def copy(self):
        ''' Make deep copy (no shared memory) of this object.
        '''
        return copy.deepcopy(self)

    def getHandleCalcLocalParams(self):
        return self.calc_local_params

    def getHandleCalcSummaryStats(self):
        return self.get_global_suff_stats
