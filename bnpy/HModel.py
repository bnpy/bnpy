'''
HModel.py

Class for representing hierarchical Bayesian models in bnpy.

Attributes
-------
allocModel : bnpy.allocmodel.AllocModel subclass
             model for generating latent cluster assignments

obsModel : bnpy.obsmodel.ObsCompModel subclass
           model for generating observed data given cluster assignments

Key functions
-------
* calc_local_params
* get_global_suff_stats
* update_global_params
'''

import numpy as np
import os
import copy

import init
from allocmodel import AllocModelConstructorsByName
from obsmodel import ObsModelConstructorsByName
from bnpy.learnalg import ElapsedTimeLogger

class HModel(object):

    def __init__(self, allocModel, obsModel):
        ''' Constructor assembles HModel given fully valid subcomponents
        '''
        self.allocModel = allocModel
        self.obsModel = obsModel
        self.inferType = allocModel.inferType
        self.initParams = None
        if hasattr(obsModel, 'setupWithAllocModel'):
            # Tell the obsModel whether to model docs or words
            obsModel.setupWithAllocModel(allocModel)

    @classmethod
    def CreateEntireModel(cls, inferType, allocModelName, obsModelName,
                          allocPriorDict, obsPriorDict, Data):
        ''' Constructor assembles HModel and all its submodels in one call
        '''
        AllocConstr = AllocModelConstructorsByName[allocModelName]
        allocModel = AllocConstr(inferType, allocPriorDict)
        ObsConstr = ObsModelConstructorsByName[obsModelName]
        obsModel = ObsConstr(inferType, Data=Data, **obsPriorDict)
        return cls(allocModel, obsModel)

    def copy(self):
        ''' Create a clone of this object with distinct memory allocation
            Any manipulation of clone's parameters will NOT affect self
        '''
        copyObj = copy.deepcopy(self)
        copyObj.obsModel.ClearCache()
        return copyObj

    def calc_local_params(self, Data, LP=None, 
            doLogElapsedTime=False, **kwargs):
        ''' Calculate local parameters specific to each data item.

            This is the E-step of the EM algorithm.
        '''
        if LP is None:
            LP = dict()
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('local', 'obsupdate')
        # Calculate  "soft evidence" each component has for each item
        # Fills in LP['E_log_soft_ev'], N x K array
        LP = self.obsModel.calc_local_params(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'obsupdate')
            ElapsedTimeLogger.startEvent('local', 'allocupdate')            
        # Combine with allocModel probs of each cluster
        # Fills in LP['resp'], N x K array whose rows sum to one
        LP = self.allocModel.calc_local_params(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'allocupdate')
        return LP

    def get_global_suff_stats(self, Data, LP, 
            doLogElapsedTime=False,
            **kwargs):
        ''' Calculate sufficient statistics for each component.

        These stats summarize the data and local parameters
        assigned to each component.

        This is necessary prep for the Global Step update.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('local', 'allocsummary')
        SS = self.allocModel.get_global_suff_stats(Data, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'allocsummary')
            ElapsedTimeLogger.startEvent('local', 'obssummary')            
        SS = self.obsModel.get_global_suff_stats(Data, SS, LP, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('local', 'obssummary')
        return SS

    def update_global_params(self, SS, rho=None, 
            doLogElapsedTime=False,
            **kwargs):
        ''' Update (in-place) global parameters given provided suff stats.
            This is the M-step of EM.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('global', 'alloc')
        self.allocModel.update_global_params(SS, rho, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'alloc')
            ElapsedTimeLogger.startEvent('global', 'obs')            
        self.obsModel.update_global_params(SS, rho, **kwargs)
        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'obs')

    def set_global_params(self, **kwargs):
        self.allocModel.set_global_params(**kwargs)
        self.obsModel.set_global_params(**kwargs)

    def insert_global_params(self, **kwargs):
        self.allocModel.insert_global_params(**kwargs)
        self.obsModel.insert_global_params(**kwargs)

    def reorderComps(self, order):
        self.allocModel.reorderComps(order)
        self.obsModel.reorderComps(order)

    def calc_evidence(self, Data=None, SS=None, LP=None,
                      scaleFactor=None, todict=False,
                      doLogElapsedTime=False, **kwargs):
        ''' Compute evidence lower bound (ELBO) objective function.
        '''
        if doLogElapsedTime:
            ElapsedTimeLogger.startEvent('global', 'ev')

        if Data is not None and LP is None and SS is None:
            LP = self.calc_local_params(Data, **kwargs)
            SS = self.get_global_suff_stats(Data, LP)
        evA = self.allocModel.calc_evidence(
            Data, SS, LP, todict=todict, **kwargs)
        evObs = self.obsModel.calc_evidence(
            Data, SS, LP, todict=todict, **kwargs)
        if scaleFactor is None:
            if hasattr(SS, 'scaleFactor'):
                scaleFactor = SS.scaleFactor
            else:
                scaleFactor = self.obsModel.getDatasetScale(SS)

        if doLogElapsedTime:
            ElapsedTimeLogger.stopEvent('global', 'ev')

        if todict:
            evA.update(evObs)
            for key in evA:
                evA[key] /= scaleFactor
            # Identify unique keys, ignoring subdivided terms
            # eg Lalloc_top_term1 and Lalloc_top_term2 are not counted,
            # since we expect they are already aggregated in term Lalloc
            ukeys = list(set([key.split('_')[0] for key in evA.keys()]))
            evA['Ltotal'] = sum([evA[key] for key in ukeys])
            return evA
        else:
            return (evA + evObs) / scaleFactor

    def calcLogLikCollapsedSamplerState(self, SS):
        ''' Compute marginal likelihood of current sampler state.
        '''
        return self.obsModel.calcMargLik(SS) \
            + self.allocModel.calcMargLik(SS)

    def init_global_params(self, Data, **initArgs):
        ''' Initialize (in-place) global parameters

            Keyword Args
            -------
            K : number of components
            initname : string name of routine for initialization
        '''
        initname = initArgs['initname']
        if 'initLP' not in initArgs:
            initArgs['initLP'] = None

        if initname.count(os.path.sep) > 0:
            init.FromSaved.init_global_params(self, Data, **initArgs)
        elif initname.count('true') > 0:
            init.FromTruth.init_global_params(self, Data, **initArgs)
        elif initname.count('LP') > 0:
            init.FromLP.init_global_params(self, Data, **initArgs)
        elif initname.count('bregmanmixture') > 0:
            init.FromScratchBregmanMixture.init_global_params(
                self, Data, **initArgs)
        elif initname.count('bregman') > 0:
            init.FromScratchBregman.init_global_params(
                self, Data, **initArgs)
        else:
            # Set hmodel global parameters "from scratch", in two stages
            # First, init obsmodel in likelihood-specific, data-driven fashion
            # Second, init allocmodel using fixed clustering from stage one.
            if str(type(self.obsModel)).count('Gauss') > 0:
                init.FromScratchGauss.init_global_params(self.obsModel,
                                                         Data, **initArgs)
            elif str(type(self.obsModel)).count('Mult') > 0:
                init.FromScratchMult.init_global_params(self.obsModel,
                                                        Data, **initArgs)
            elif str(type(Data)).count('Graph') > 0:
                initArgs['initLP'] = \
                    init.FromScratchRelational.init_global_params(
                        self.obsModel, Data, **initArgs)
            elif str(type(self.obsModel)).count('Bern') > 0:
                init.FromScratchBern.init_global_params(self.obsModel,
                                                        Data, **initArgs)
            else:
                raise NotImplementedError('Unrecognized initname procedure.')

            if 'K' in initArgs:
                # Make sure K is exactly same for both alloc and obs models
                # Needed because obsModel init can sometimes yield K < Kinput
                initArgs['K'] = self.obsModel.K
            self.allocModel.init_global_params(Data=Data, **initArgs)

    def getAllocModelName(self):
        return self.allocModel.__class__.__name__

    def getObsModelName(self):
        return self.obsModel.__class__.__name__

    def get_model_info(self):
        s = 'Allocation Model:  %s\n' % (self.allocModel.get_info_string())
        s += 'Obs. Data  Model:  %s\n' % (self.obsModel.get_info_string())
        s += 'Obs. Data  Prior:  %s' % (self.obsModel.get_info_string_prior())
        return s

    def getBestMergePair(self, SS, mPairIDs):
        """ Identify best merge pair among given list of candidates.

        Here, best is measured by the pair that would improve ELBO most.

        Returns
        -------
        mPair : tuple of (kA, kB)
        """
        aGap = self.allocModel.calcHardMergeGap_SpecificPairs(SS, mPairIDs)
        oGap = self.obsModel.calcHardMergeGap_SpecificPairs(SS, mPairIDs)
        gap = aGap + oGap
        bestid = gap.argmax()
        return mPairIDs[bestid]
