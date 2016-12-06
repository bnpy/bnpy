import numpy as np
import BLogger

from bnpy.allocmodel.mix.DPMixtureRestrictedLocalStep import \
    summarizeRestrictedLocalStep_DPMixtureModel, \
    makeExpansionSSFromZ_DPMixtureModel

from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep2 import \
    summarizeRestrictedLocalStep_HDPTopicModel

from bnpy.allocmodel.topics.HDPTopicRestrictedLocalStep import \
    makeExpansionSSFromZ_HDPTopicModel
    #summarizeRestrictedLocalStep_HDPTopicModel

# Load custom function for each possible allocation model
RestrictedLocalStepFuncMap = dict(
    DPMixtureModel=summarizeRestrictedLocalStep_DPMixtureModel,
    HDPTopicModel=summarizeRestrictedLocalStep_HDPTopicModel,
    )

MakeSSFromZFuncMap = dict(
    DPMixtureModel=makeExpansionSSFromZ_DPMixtureModel,
    HDPTopicModel=makeExpansionSSFromZ_HDPTopicModel,
    )

def summarizeRestrictedLocalStep(
        Dslice=None, curModel=None, curLPslice=None, **kwargs):
    global RestrictedLocalStepFuncMap
    allocModelName = curModel.getAllocModelName()
    if allocModelName not in RestrictedLocalStepFuncMap:
        raise NotImplementedError('Restricted local step function for ' + \
            allocModelName)
    xSSslice = RestrictedLocalStepFuncMap[allocModelName](
        Dslice=Dslice,
        curModel=curModel,
        curLPslice=curLPslice, **kwargs)
    return xSSslice

def makeExpansionSSFromZ(
        Dslice=None, curModel=None, curLPslice=None,
        **kwargs):
    global MakeSSFromZFuncMap
    allocModelName = curModel.getAllocModelName()
    if allocModelName not in MakeSSFromZFuncMap:
        raise NotImplementedError('makeExpansionSSFromZ function for ' + \
            allocModelName)
    xSSslice = MakeSSFromZFuncMap[allocModelName](
        Dslice=Dslice, curModel=curModel, curLPslice=curLPslice,
        **kwargs)
    return xSSslice    
