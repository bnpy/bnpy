'''
ModelReader.py

Load bnpy models from disk

See Also
-------
ModelWriter.py : save bnpy models to disk.
'''
import numpy as np
import scipy.io
import os
import glob

from ModelWriter import makePrefixForLap
from bnpy.allocmodel import AllocModelConstructorsByName
from bnpy.obsmodel import ObsModelConstructorsByName
from bnpy.util import toCArray, as1D, as2D


def getPrefixForLapQuery(taskpath, lapQuery):
    ''' Search among checkpoint laps for one nearest to query.

    Returns
    --------
    prefix : str
        For lap 1, prefix = 'Lap0001.000'.
        For lap 5.5, prefix = 'Lap0005.500'.
    lap : int
        lap checkpoint for saved params close to lapQuery
    '''
    try:
        saveLaps = np.loadtxt(os.path.join(taskpath, 'snapshot_lap.txt'))
    except IOError:
        fileList = glob.glob(os.path.join(taskpath, 'Lap*Topic*'))
        if len(fileList) == 0:
            fileList = glob.glob(os.path.join(taskpath, 'Lap*.log_prob_w'))
        assert len(fileList) > 0
        saveLaps = list()
        for fpath in sorted(fileList):
            basename = fpath.split(os.path.sep)[-1]
            lapstr = basename[3:11]
            saveLaps.append(float(lapstr))
        saveLaps = np.sort(np.asarray(saveLaps))

    saveLaps = as1D(saveLaps)
    if lapQuery is None:
        bestLap = saveLaps[-1]  # take final saved value
    else:
        distances = np.abs(lapQuery - saveLaps)
        bestLap = saveLaps[np.argmin(distances)]
    return makePrefixForLap(bestLap), bestLap


def load_model_at_lap(matfilepath, lapQuery):
    ''' Loads saved model with lap closest to provided lapQuery.

    Returns
    -------
    model : bnpy.HModel
        Model object for saved at checkpoint lap=bestLap.
    bestLap : int
        lap checkpoint for saved model closed to lapQuery
    '''
    prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
    model = load_model_at_prefix(matfilepath, prefix=prefix)
    return model, bestLap


def load_model_at_prefix(matfilepath, prefix='Best', lap=None):
    ''' Load model stored to disk by ModelWriter

    Returns
    ------
    model : bnpy.HModel
        Model object for saved at checkpoint indicated by prefix or lap.
    '''
    # Avoids circular import
    import bnpy.HModel as HModel

    if lap is not None:
        prefix, _ = getPrefixForLapQuery(matfilepath, lap)
    try:
        obsModel = load_obs_model(matfilepath, prefix)
        allocModel = load_alloc_model(matfilepath, prefix)
        model = HModel(allocModel, obsModel)
    except IOError as e:
        print str(e)
        '''
        if prefix == 'Best':
            matList = glob.glob(os.path.join(matfilepath, '*TopicModel.mat'))
            lpwList = glob.glob(os.path.join(matfilepath, '*.log_prob_w'))
            if len(matList) > 0:
                matList.sort()  # ascending order, so most recent is last
                prefix = matList[-1].split(os.path.sep)[-1][:11]
                model = loadTopicModel(matfilepath, prefix=prefix)
            elif len(lpwList) > 0:
                lpwList.sort()  # ascenting order
                prefix = lpwList[-1].split(os.path.sep)[-1][:7]

            else:
                raise e
        '''
        try:
            model = loadTopicModel(matfilepath, prefix=prefix)
        except IOError as e:
            model = loadTopicModelFromMEDLDA(matfilepath, prefix=prefix)
    return model


def load_alloc_model(matfilepath, prefix):
    """ Load allocmodel stored to disk in bnpy .mat format.

    Parameters
    ------
    matfilepath : str
        String file system path to folder where .mat files are stored.
        Usually this path is a "taskoutpath" like where bnpy.run
        saves its output.
    prefix : str
        Indicates which stored checkpoint to use.
        Can look like 'Lap0005.000'.

    Returns
    ------
    allocModel : bnpy.allocmodel object
        This object has valid set of global parameters
        and valid hyperparameters that define its prior.
    """
    apriorpath = os.path.join(matfilepath, 'AllocPrior.mat')
    amodelpath = os.path.join(matfilepath, prefix + 'AllocModel.mat')
    APDict = loadDictFromMatfile(apriorpath)
    ADict = loadDictFromMatfile(amodelpath)
    AllocConstr = AllocModelConstructorsByName[ADict['name']]
    amodel = AllocConstr(ADict['inferType'], APDict)
    amodel.from_dict(ADict)
    return amodel


def load_obs_model(matfilepath, prefix):
    """ Load observation model object stored to disk in bnpy mat format.

    Parameters
    ------
    matfilepath : str
        String file system path to folder where .mat files are stored.
        Usually this path is a "taskoutpath" like where bnpy.run
        saves its output.
    prefix : str
        Indicates which stored checkpoint to use.
        Can look like 'Lap0005.000'.

    Returns
    ------
    allocModel : bnpy.allocmodel object
        This object has valid set of global parameters
        and valid hyperparameters that define its prior.
    """
    obspriormatfile = os.path.join(matfilepath, 'ObsPrior.mat')
    PriorDict = loadDictFromMatfile(obspriormatfile)
    ObsConstr = ObsModelConstructorsByName[PriorDict['name']]
    obsModel = ObsConstr(**PriorDict)

    obsmodelpath = os.path.join(matfilepath, prefix + 'ObsModel.mat')
    ParamDict = loadDictFromMatfile(obsmodelpath)
    if obsModel.inferType == 'EM':
        obsModel.setEstParams(**ParamDict)
    else:
        obsModel.setPostFactors(**ParamDict)
    return obsModel


def loadDictFromMatfile(matfilepath):
    ''' Load dict of numpy arrays from a .mat-format file on disk.

    This is a wrapper around scipy.io.loadmat,
    which makes the returned numpy arrays in standard aligned format.

    Returns
    --------
    D : dict
        Each key/value pair is a parameter name and a numpy array
        loaded from the provided mat file.
        We ensure before returning that each array has properties:
            * C alignment
            * Original 2D shape has been squeezed as much as possible
                * (1,1) becomes a size=1 1D array
                * (1,N) or (N,1) become 1D arrays
            * flags.aligned is True
            * flags.owndata is True
            * dtype.byteorder is '='

    Examples
    -------
    >>> import scipy.io
    >>> Dorig = dict(scalar=5, scalar1DN1=np.asarray([3.14,]))
    >>> Dorig['arr1DN3'] = np.asarray([1,2,3])
    >>> scipy.io.savemat('Dorig.mat', Dorig, oned_as='row')
    >>> D = loadDictFromMatfile('Dorig.mat')
    >>> D['scalar']
    array(5)
    >>> D['scalar1DN1']
    array(3.14)
    >>> D['arr1DN3']
    array([1, 2, 3])
    '''
    Dtmp = scipy.io.loadmat(matfilepath)
    D = dict([x for x in Dtmp.items() if not x[0].startswith('__')])
    for key in D:
        if not isinstance(D[key], np.ndarray):
            continue
        x = D[key]
        if isinstance(x[0], np.unicode_):
            if x.size == 1:
                D[key] = str(x[0])
            else:
                D[key] = tuple([str(s) for s in x])
            continue
        if x.ndim == 2:
            x = np.squeeze(x)
        if str(x.dtype).count('int'):
            arr = toCArray(x, dtype=np.int32)
        else:
            arr = toCArray(x, dtype=np.float64)
        assert arr.dtype.byteorder == '='
        assert arr.flags.aligned is True
        assert arr.flags.owndata is True
        D[key] = arr
    return D


def loadWordCountMatrixForLap(matfilepath, lapQuery, toDense=True):
    ''' Load word counts
    '''
    prefix, bestLap = getPrefixForLapQuery(matfilepath, lapQuery)
    _, WordCounts = loadTopicModel(matfilepath,
        prefix=prefix, returnWordCounts=1)
    return WordCounts


def loadTopicModelFromMEDLDA(filepath,
                             prefix=None,
                             returnTPA=0):
    ''' Load topic model saved in medlda format.
    '''
    # Avoid circular import
    import bnpy.HModel as HModel

    assert prefix is not None
    alphafilepath = os.path.join(filepath, prefix + '.alpha')
    etafilepath = os.path.join(filepath, prefix + '.eta')
    topicfilepath = os.path.join(filepath, prefix + '.log_prob_w')

    alpha = float(np.loadtxt(alphafilepath))
    eta = np.loadtxt(etafilepath)
    logtopics = np.loadtxt(topicfilepath)
    topics = np.exp(logtopics)
    topics += 1e-9
    topics /= topics.sum(axis=1)[:, np.newaxis]
    assert np.all(np.isfinite(topics))

    if returnTPA:
        K = topics.shape[0]
        probs = 1.0 / K * np.ones(K)
        return topics, probs, alpha, eta

    infAlg = 'VB'
    aPriorDict = dict(alpha=alpha)
    amodel = AllocModelConstructorsByName[
        'FiniteTopicModel'](infAlg, aPriorDict)
    omodel = ObsModelConstructorsByName['Mult'](infAlg,
                                                lam=0.001, D=topics.shape[1])
    hmodel = HModel(amodel, omodel)
    hmodel.obsModel.set_global_params(topics=topics, nTotalTokens=1000)
    return hmodel


def loadTopicModel(
        matfilepath, 
        queryLap=None,
        prefix=None,
        returnWordCounts=0,
        returnTPA=0, 
        normalizeTopics=0,
        normalizeProbs=0,
        **kwargs):
    ''' Load saved topic model

    Returns
    -------
    topics : 2D array, K x vocab_size (if returnTPA)
    probs : 1D array, size K (if returnTPA)
    alpha : scalar (if returnTPA)
    hmodel : HModel
    WordCounts : 2D array, size K x vocab_size (if returnWordCounts)
    '''
    if prefix is None:
        prefix, lapQuery = getPrefixForLapQuery(matfilepath, queryLap)
    # avoids circular import
    from bnpy.HModel import HModel
    if len(glob.glob(os.path.join(matfilepath, "*.log_prob_w"))) > 0:
        return loadTopicModelFromMEDLDA(matfilepath, prefix,
                                        returnTPA=returnTPA)

    snapshotList = glob.glob(os.path.join(matfilepath, 'Lap*TopicSnapshot'))
    matfileList = glob.glob(os.path.join(matfilepath, 'Lap*TopicModel.mat'))
    if len(snapshotList) > 0:
        if prefix is None:
            snapshotList.sort()
            snapshotPath = snapshotList[-1]
        else:
            snapshotPath = None
            for curPath in snapshotList:
                if curPath.count(prefix):
                    snapshotPath = curPath
        return loadTopicModelFromTxtFiles(
            snapshotPath,
            normalizeTopics=normalizeTopics,
            normalizeProbs=normalizeProbs,
            returnWordCounts=returnWordCounts,
            returnTPA=returnTPA)

    if prefix is not None:
        matfilepath = os.path.join(matfilepath, prefix + 'TopicModel.mat')
    Mdict = loadDictFromMatfile(matfilepath)
    if 'SparseWordCount_data' in Mdict:
        data = np.asarray(Mdict['SparseWordCount_data'], dtype=np.float64)
        K = int(Mdict['K'])
        vocab_size = int(Mdict['vocab_size'])
        try:
            indices = Mdict['SparseWordCount_indices']
            indptr = Mdict['SparseWordCount_indptr']
            WordCounts = scipy.sparse.csr_matrix((data, indices, indptr),
                                                 shape=(K, vocab_size))
        except KeyError:
            rowIDs = Mdict['SparseWordCount_i'] - 1
            colIDs = Mdict['SparseWordCount_j'] - 1
            WordCounts = scipy.sparse.csr_matrix((data, (rowIDs, colIDs)),
                                                 shape=(K, vocab_size))
        Mdict['WordCounts'] = WordCounts.toarray()
    if returnTPA:
        # Load topics : 2D array, K x vocab_size
        if 'WordCounts' in Mdict:
            topics = Mdict['WordCounts'] + Mdict['lam']
        else:
            topics = Mdict['topics']
        topics = as2D(toCArray(topics, dtype=np.float64))
        assert topics.ndim == 2
        K = topics.shape[0]
        if normalizeTopics:
            topics /= topics.sum(axis=1)[:,np.newaxis]

        # Load probs : 1D array, size K
        try:
            probs = Mdict['probs']
        except KeyError:
            probs = (1.0 / K) * np.ones(K)
        probs = as1D(toCArray(probs, dtype=np.float64))
        assert probs.ndim == 1
        assert probs.size == K
        if normalizeProbs:
            probs = probs / np.sum(probs)

        # Load alpha : scalar float > 0
        try:
            alpha = float(Mdict['alpha'])
        except KeyError:
            if 'alpha' in os.environ:
                alpha = float(os.environ['alpha'])
            else:
                raise ValueError('Unknown parameter alpha')
        if 'eta' in Mdict:
            return topics, probs, alpha, as1D(toCArray(Mdict['eta']))
        return topics, probs, alpha

    infAlg = 'VB'
    if 'gamma' in Mdict:
        aPriorDict = dict(alpha=Mdict['alpha'], gamma=Mdict['gamma'])
        HDPTopicModel = AllocModelConstructorsByName['HDPTopicModel']
        amodel = HDPTopicModel(infAlg, aPriorDict)
    else:
        FiniteTopicModel = AllocModelConstructorsByName['FiniteTopicModel']
        amodel = FiniteTopicModel(infAlg, dict(alpha=Mdict['alpha']))
    omodel = ObsModelConstructorsByName['Mult'](infAlg, **Mdict)
    hmodel = HModel(amodel, omodel)
    hmodel.set_global_params(**Mdict)
    if returnWordCounts:
        return hmodel, Mdict['WordCounts']
    return hmodel


def loadTopicModelFromTxtFiles(
        snapshotPath, returnTPA=False, returnWordCounts=False,
        normalizeProbs=True,
        normalizeTopics=True,
        **kwargs):
    ''' Load from snapshot text files.

    Returns
    -------
    hmodel
    '''
    Mdict = dict()
    possibleKeys = ['K', 'probs', 'alpha', 'beta', 'lam',
        'gamma', 'nTopics', 'nTypes', 'vocab_size']
    keyMap = dict(beta='lam', nTopics='K', nTypes='vocab_size')
    for key in possibleKeys:
        try:
            arr = np.loadtxt(snapshotPath + "/%s.txt" % (key))
            if key in keyMap:
                Mdict[keyMap[key]] = arr
            else:
                Mdict[key] = arr
        except Exception:
            pass
    assert 'K' in Mdict
    assert 'lam' in Mdict
    K = int(Mdict['K'])
    V = int(Mdict['vocab_size'])

    if os.path.exists(snapshotPath + "/topics.txt"):
        Mdict['topics'] = np.loadtxt(snapshotPath + "/topics.txt")
        Mdict['topics'] = as2D(toCArray(Mdict['topics'], dtype=np.float64))
        assert Mdict['topics'].ndim == 2
        assert Mdict['topics'].shape == (K,V)
    else:
        TWC_data = np.loadtxt(snapshotPath + "/TopicWordCount_data.txt")
        TWC_inds = np.loadtxt(
            snapshotPath + "/TopicWordCount_indices.txt", dtype=np.int32)
        if os.path.exists(snapshotPath + "/TopicWordCount_cscindptr.txt"):
            TWC_cscindptr = np.loadtxt(
                snapshotPath + "/TopicWordCount_cscindptr.txt", dtype=np.int32)
            TWC = scipy.sparse.csc_matrix(
                (TWC_data, TWC_inds, TWC_cscindptr), shape=(K,V))
        else:
            TWC_csrindptr = np.loadtxt(
                snapshotPath + "/TopicWordCount_indptr.txt", dtype=np.int32)
            TWC = scipy.sparse.csr_matrix(
                (TWC_data, TWC_inds, TWC_csrindptr), shape=(K,V))

        Mdict['WordCounts'] = TWC.toarray()

    if returnTPA:
        # Load topics : 2D array, K x vocab_size
        if 'WordCounts' in Mdict:
            topics = Mdict['WordCounts'] + Mdict['lam']
        else:
            topics = Mdict['topics']
        topics = as2D(toCArray(topics, dtype=np.float64))
        assert topics.ndim == 2
        K = topics.shape[0]
        if normalizeTopics:
            topics /= topics.sum(axis=1)[:,np.newaxis]

        # Load probs : 1D array, size K
        try:
            probs = Mdict['probs']
        except KeyError:
            probs = (1.0 / K) * np.ones(K)
        probs = as1D(toCArray(probs, dtype=np.float64))
        assert probs.ndim == 1
        assert probs.size == K
        if normalizeProbs:
            probs = probs / np.sum(probs)

        # Load alpha : scalar float > 0
        try:
            alpha = float(Mdict['alpha'])
        except KeyError:
            if 'alpha' in os.environ:
                alpha = float(os.environ['alpha'])
            else:
                raise ValueError('Unknown parameter alpha')
        return topics, probs, alpha

    # BUILD HMODEL FROM LOADED TXT
    infAlg = 'VB'
    # avoids circular import
    from bnpy.HModel import HModel
    if 'gamma' in Mdict:
        aPriorDict = dict(alpha=Mdict['alpha'], gamma=Mdict['gamma'])
        HDPTopicModel = AllocModelConstructorsByName['HDPTopicModel']
        amodel = HDPTopicModel(infAlg, aPriorDict)
    else:
        FiniteTopicModel = AllocModelConstructorsByName['FiniteTopicModel']
        amodel = FiniteTopicModel(infAlg, dict(alpha=Mdict['alpha']))
    omodel = ObsModelConstructorsByName['Mult'](infAlg, **Mdict)
    hmodel = HModel(amodel, omodel)
    hmodel.set_global_params(**Mdict)
    if returnWordCounts:
        return hmodel, Mdict['WordCounts']
    return hmodel
