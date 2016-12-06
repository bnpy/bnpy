'''
FromSaved.py

Initialize params of a bnpy model from a previous result saved to disk.
'''
import numpy as np
import scipy.io
import os
from bnpy.ioutil import ModelReader


def init_global_params(hmodel, Data, initname=None, **kwargs):
    ''' Initialize (in-place) the global params of the given hmodel.

    Copies parameters stored to disk from a previous run.

    Only global parameters are modified.
    This does NOT alter settings of hmodel's prior distribution.

    Parameters
    -------
    hmodel : bnpy.HModel
        model object to initialize
    Data   : bnpy.data.DataObj
         Dataset to use to drive initialization.
         hmodel.obsModel dimensions must match this dataset.
    initname : str
        valid filesystem path to stored result

    Post Condition
    -------
    hmodel has valid global parameters.
    '''
    if os.path.isdir(initname):
        try:
            # First try assumes initname contains jobname and taskid
            init_global_params_from_bnpy_format(
                hmodel, Data, initname, **kwargs)
        except:
            # Second tacks on taskid, if initname contains only jobname
            initname2 = os.path.join(initname, str(kwargs['taskid']))
            init_global_params_from_bnpy_format(
                hmodel, Data, initname2, **kwargs)
    elif initname.count('.mat') > 0:
        # Handle external external formats (not bnpy models) saved as MAT file
        MatDict = scipy.io.loadmat(initname)
        hmodel.set_global_params(**MatDict)
    else:
        raise ValueError('Unrecognized init file: %s' % (initname))


def init_global_params_from_bnpy_format(hmodel, Data, initname,
                                        initLapFrac=-1,
                                        prefix='Best', **kwargs):
    """ Initialize global parameters for hmodel from bnpy disk format.

    Post Condition
    -------
    hmodel has valid global parameters.
    '''
    """
    if initLapFrac > -1:
        storedModel, lap = ModelReader.loadModelForLap(initname, initLapFrac)
    else:
        storedModel = ModelReader.load_model(initname, prefix)
    try:
        hmodel.set_global_params(hmodel=storedModel,
                                 obsModel=storedModel.obsModel)
    except AttributeError:
        LP = storedModel.calc_local_params(Data)
        SS = hmodel.get_global_suff_stats(Data, LP)
        hmodel.update_global_params(SS)
