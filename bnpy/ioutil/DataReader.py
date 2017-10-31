from builtins import *
import os
import numpy as np
import bnpy.data

from bnpy.data.DataIterator import DataIterator

def loadDataFromSavedTask(taskoutpath, dataSplitName='train', **kwargs):
    ''' Load data object used for training a specified saved run.

    Args
    ----
    taskoutpath : full path to saved results of bnpy training run

    Returns
    -------
    Dslice : bnpy Data object
        used for training the specified task.

    Example
    -------
    >>> import bnpy
    >>> os.environ['BNPYOUTDIR'] = '/tmp/'
    >>> hmodel, Info = bnpy.run(
    ...     'AsteriskK8', 'FiniteMixtureModel', 'Gauss', 'VB',
    ...     nLap=1, nObsTotal=144, K=10,
    ...     doWriteStdOut=False)
    >>> outputdir = Info['outputdir']
    >>> Data2 = loadDataFromSavedTask(outputdir)
    >>> print Data2.nObsTotal
    144
    >>> np.allclose(Info['Data'].X, Data2.X)
    True
    '''
    dataName = getDataNameFromTaskpath(taskoutpath)
    dataKwargs = loadDataKwargsFromDisk(taskoutpath)
    try:
        onlineKwargs = loadKwargsFromDisk(
            taskoutpath, 'args-OnlineDataPrefs.txt')
        dataKwargs.update(onlineKwargs)
    except IOError:
        pass
    except ValueError:
        # Occurs if does not exist.
        pass

    try:
        datamod = __import__(dataName, fromlist=[])
        if dataSplitName.count('test'):
            Data = datamod.get_test_data(**dataKwargs)
        elif dataSplitName.count('valid'):
            Data = datamod.get_validation_data(**dataKwargs)
        else:
            Data = datamod.get_data(**dataKwargs)
            if 'nBatch' in dataKwargs and dataKwargs['nBatch'] > 1:
                DI = DataIterator(Data, alwaysTrackTruth=1, **dataKwargs)
                if 'batchID' in kwargs and kwargs['batchID'] is not None:
                    batchID = kwargs['batchID']
                else:
                    batchID = 0
                Dbatch = DI.getBatch(batchID)
                Dbatch.name = Data.name
                return Dbatch
        return Data

    except ImportError as e:
        Data = None
        if 'dataPath' not in dataKwargs:
            raise e
        if dataSplitName == 'train':
            # Load from file
            DI = bnpy.data.DataIteratorFromDisk(**dataKwargs)
            if 'batchID' in kwargs and kwargs['batchID'] is not None:
                Data = DI.getBatch(kwargs['batchID'])
            else:
                Data = DI.getBatch(0)
            Data.name = dataName
        elif dataSplitName == 'test':
            try:
                confpath = os.path.join(dataKwargs['dataPath'], 'Info.conf')
                if os.path.exists(confpath):
                    with open(confpath, 'r') as f:
                        for line in f.readlines():
                            if line.count('heldoutDataPath'):
                                matpath = line.split('=')[1]
                                matpath = matpath.strip()
                matpath = os.path.expandvars(matpath)
                if os.path.exists(matpath) and matpath.endswith('.mat'):
                    if 'vocab_size' in dataKwargs:
                        Data = bnpy.data.BagOfWordsData.LoadFromFile_tokenlist(
                            matpath, vocab_size=dataKwargs['vocab_size'])
                        Data.name = dataName
                    else:
                        Data = bnpy.data.GroupXData.LoadFromFile(matpath)
                        Data.name = dataName
            except Exception as e2:
                raise e
        return Data

def loadDataKwargsFromDisk(taskoutpath):
    ''' Load keyword options used to load specific dataset.

    Returns
    -------
    dataKwargs : dict with options for loading dataset
    '''
    return loadKwargsFromDisk(taskoutpath, 'args-DatasetPrefs.txt')

def loadKwargsFromDisk(taskoutpath,
        txtfile='args-birth.txt',
        suffix=None):
    ''' Load keyword options from specified txtfile.

    Returns
    -------
    kwargs : dict
    '''
    txtfilepath = os.path.join(taskoutpath, txtfile)
    if not os.path.exists(txtfilepath):
        raise ValueError("Cannot find options text file:\n %s" % (
            txtfilepath))
    kwargs = dict()
    with open(txtfilepath, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split(' ')
            assert len(fields) == 2
            if suffix:
                if fields[0].endswith(suffix):
                    kwargs[fields[0]] = str2numorstr(fields[1])
            else:
                kwargs[fields[0]] = str2numorstr(fields[1])
    return kwargs

def loadLPKwargsFromDisk(taskoutpath):
    ''' Load keyword options used to load specific dataset.

    Returns
    -------
    dataKwargs : dict with options for loading dataset
    '''
    from bnpy.ioutil.BNPYArgParser import algChoices
    chosentxtfile = None
    for algName in algChoices:
        txtfile = 'args-%s.txt' % (algName)
        txtfilepath = os.path.join(taskoutpath, txtfile)
        if os.path.exists(txtfilepath):
            chosentxtfile = txtfile
            break
    if chosentxtfile is None:
        raise ValueError("No args options file found.")
    LPkwargs = loadKwargsFromDisk(taskoutpath, chosentxtfile, suffix='LP')
    return LPkwargs

def str2numorstr(s):
    ''' Convert string to numeric type if possible.

    Returns
    -------
    val : int or float or str
    '''
    try:
        val = int(s)
    except ValueError as e:
        try:
            val = float(s)
        except ValueError as e:
            val = str(s)
    return val

def getDataNameFromTaskpath(taskoutpath):
    ''' Extract dataset name from bnpy output filepath.

    Returns
    -------
    dataName : string identifier of a dataset

    Examples
    --------
    >>> os.environ['BNPYOUTDIR'] = '/tmp/'
    >>> taskoutpath = '/tmp/MyDataName/myjobname/1/'
    >>> dataName = getDataNameFromTaskpath(taskoutpath)
    >>> print dataName
    MyDataName
    '''
    # Make it a proper absolute path
    taskoutpath = os.path.abspath(taskoutpath)
    # Extract the dataset name from taskoutpath
    strippedpath = taskoutpath.replace(os.environ['BNPYOUTDIR'], '')
    if strippedpath.startswith(os.path.sep):
        strippedpath = strippedpath[1:]
    # The very next segment must be the data name
    dataName = strippedpath[:strippedpath.index(os.path.sep)]
    return dataName
