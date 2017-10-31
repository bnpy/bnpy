'''
Classes
-----
GroupXData
    Data object for holding a dense matrix X of real 64-bit floats,
    organized contiguously based on provided group structure.
'''
from builtins import *
import numpy as np
from collections import namedtuple

from .XData import XData
from bnpy.util import as1D, as2D, as3D, toCArray
from bnpy.util import numpyToSharedMemArray, sharedMemToNumpyArray


class GroupXData(XData):

    """ Dataset object for dense real vectors organized in groups.

    GroupXData can represent situations like:
    * obseved image patches, across many images
        group=image, observation=patch
    * observed test results for patients, across many hospitals
        group=hospital, obsevation=patient test result

    Attributes
    ------
    X : 2D array, size N x D
        each row is a single dense observation vector
    Xprev : 2D array, size N x D, optional
        "previous" observations for auto-regressive likelihoods
    dim : int
        the dimension of each observation
    nObs : int
        the number of in-memory observations for this instance
    TrueParams : dict
        key/value pairs represent names and arrays of true parameters
    doc_range : 1D array, size nDoc+1
        the number of in-memory observations for this instance
    nDoc : int
        the number of in-memory documents for this instance
    nDocTotal : int
        total number of documents in entire dataset

    Example
    --------
    # Create 1000 observations, each one a 3D vector
    >>> X = np.random.randn(1000, 3)

    # Assign items 0-499 to doc 1, 500-1000 to doc 2
    >>> doc_range = [0, 500, 1000]
    >>> myData = GroupXData(X, doc_range)
    >>> print myData.nObs
    1000
    >>> print myData.X.shape
    (1000, 3)
    >>> print myData.nDoc
    2
    """
    @classmethod
    def LoadFromFile(cls, filepath, nDocTotal=None, **kwargs):
        ''' Constructor for loading data from disk into XData instance
        '''
        if filepath.endswith('.mat'):
            return cls.read_mat(filepath, nDocTotal=nDocTotal, **kwargs)
        raise NotImplemented('Only .mat file supported.')

    def save_to_mat(self, matfilepath):
        ''' Save contents of current object to disk
        '''
        import scipy.io
        SaveVars = dict(X=self.X, nDoc=self.nDoc, doc_range=self.doc_range)
        if hasattr(self, 'Xprev'):
            SaveVars['Xprev'] = self.Xprev
        if hasattr(self, 'TrueParams') and 'Z' in self.TrueParams:
            SaveVars['TrueZ'] = self.TrueParams['Z']
        scipy.io.savemat(matfilepath, SaveVars, oned_as='row')

    @classmethod
    def read_npz(cls, npzfilepath, nDocTotal=None, **kwargs):
        ''' Constructor for building an instance of GroupXData from npz
        '''
        var_dict = dict(**np.load(npzfilepath))
        if 'X' not in var_dict:
            raise KeyError(
                'Stored npz file needs to have data in field named X')
        if 'doc_range' not in var_dict:
            raise KeyError(
                'Stored npz file needs to have field named doc_range')
        if nDocTotal is not None:
            var_dict['nDocTotal'] = nDocTotal
        return cls(**var_dict)

    @classmethod
    def read_mat(cls, matfilepath, nDocTotal=None, **kwargs):
        ''' Constructor for building an instance of GroupXData from disk
        '''
        import scipy.io
        InDict = scipy.io.loadmat(matfilepath)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        if 'doc_range' not in InDict:
            raise KeyError(
                'Stored matfile needs to have field named doc_range')
        if nDocTotal is not None:
            InDict['nDocTotal'] = nDocTotal
        return cls(**InDict)

    def __init__(self, X=None, doc_range=None, nDocTotal=None,
                 Xprev=None, TrueZ=None,
                 TrueParams=None, fileNames=None, summary=None, **kwargs):
        ''' Create an instance of GroupXData for provided array X

        Post Condition
        ---------
        self.X : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        self.Xprev : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        self.doc_range : 1D array, size nDoc+1
        '''
        self.X = as2D(toCArray(X, dtype=np.float64))
        self.doc_range = as1D(toCArray(doc_range, dtype=np.int32))
        if summary is not None:
            self.summary = summary
        if Xprev is not None:
            self.Xprev = as2D(toCArray(Xprev, dtype=np.float64))

        # Verify attributes are consistent
        self._set_dependent_params(doc_range, nDocTotal)
        self._check_dims()

        # Add optional true parameters / true hard labels
        if TrueParams is not None:
            self.TrueParams = dict()
            for key, arr in list(TrueParams.items()):
                self.TrueParams[key] = toCArray(arr)

        if TrueZ is not None:
            if not hasattr(self, 'TrueParams'):
                self.TrueParams = dict()
            self.TrueParams['Z'] = as1D(toCArray(TrueZ))
            self.TrueParams['K'] = np.unique(self.TrueParams['Z']).size

        # Add optional source files for each group/sequence
        if fileNames is not None:
            if hasattr(fileNames, 'shape') and fileNames.shape == (1, 1):
                fileNames = fileNames[0, 0]
            if len(fileNames) > 1:
                self.fileNames = [str(x).strip()
                                  for x in np.squeeze(fileNames)]
            else:
                self.fileNames = [str(fileNames[0])]
        # Add extra data attributes custom for the dataset
        for key in kwargs:
            if hasattr(self, key):
                continue
            if not key.startswith("__"):
                arr = np.squeeze(as1D(kwargs[key]))
                if arr.shape == ():
                    try:
                        arr = float(arr)
                    except TypeError:
                        continue
                setattr(self, key, arr)

    def _set_dependent_params(self, doc_range, nDocTotal=None):
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.nDoc = self.doc_range.size - 1
        if nDocTotal is None:
            self.nDocTotal = self.nDoc
        else:
            self.nDocTotal = int(nDocTotal)

    def _check_dims(self):
        assert self.X.ndim == 2
        assert self.X.flags.c_contiguous
        assert self.X.flags.owndata
        assert self.X.flags.aligned
        assert self.X.flags.writeable

        assert self.doc_range.ndim == 1
        assert self.doc_range.size == self.nDoc + 1
        assert self.doc_range[0] == 0
        assert self.doc_range[-1] == self.nObs
        assert np.all(self.doc_range[1:] - self.doc_range[:-1] >= 0)

    def get_size(self):
        return self.nDoc

    def get_total_size(self):
        return self.nDocTotal

    def get_dim(self):
        return self.dim

    def get_text_summary(self):
        ''' Returns human-readable description of this dataset
        '''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = 'GroupXData'
        return s

    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        s = '  size: %d units (documents)\n' % (self.get_size())
        s += '  dimension: %d' % (self.get_dim())
        return s

    def toXData(self):
        ''' Return simplified XData instance, losing group structure
        '''
        if hasattr(self, 'TrueParams'):
            TParams = self.TrueParams
        else:
            TParams=None

        if hasattr(self, 'Xprev'):
            return XData(self.X, Xprev=self.Xprev, TrueParams=TParams)
        else:
            return XData(self.X, TrueParams=TParams)

    # Create Subset
    #########################################################
    def make_subset(self,
            docMask=None,
            atomMask=None,
            doTrackTruth=False,
            doTrackFullSize=True):
        """ Get subset of this dataset identified by provided unit IDs.

        Parameters
        -------
        docMask : 1D array_like of ints
            Identifies units (documents) to use to build subset.
        doTrackFullSize : boolean, optional
            default=True
            If True, return DataObj with same nDocTotal value as this
            dataset. If False, returned DataObj has smaller size.
        atomMask : 1D array_like of ints, optional
            default=None
            If present, identifies rows of X to return as XData

        Returns
        -------
        Dchunk : bnpy.data.GroupXData instance
        """
        if atomMask is not None:
            return self.toXData().select_subset_by_mask(atomMask)

        if len(docMask) < 1:
            raise ValueError('Cannot select empty subset')

        newXList = list()
        newXPrevList = list()
        newDocRange = np.zeros(len(docMask) + 1)
        newPos = 1
        for d in range(len(docMask)):
            start = self.doc_range[docMask[d]]
            stop = self.doc_range[docMask[d] + 1]
            newXList.append(self.X[start:stop])
            if hasattr(self, 'Xprev'):
                newXPrevList.append(self.Xprev[start:stop])
            newDocRange[newPos] = newDocRange[newPos - 1] + stop - start
            newPos += 1
        newX = np.vstack(newXList)
        if hasattr(self, 'Xprev'):
            newXprev = np.vstack(newXPrevList)
        else:
            newXprev = None
        if doTrackFullSize:
            nDocTotal = self.nDocTotal
        else:
            nDocTotal = None

        if hasattr(self, 'alwaysTrackTruth'):
            doTrackTruth = doTrackTruth or self.alwaysTrackTruth
        hasTrueZ = hasattr(self, 'TrueParams') and 'Z' in self.TrueParams
        if doTrackTruth and hasTrueZ:
            TrueZ = self.TrueParams['Z']
            newTrueZList = list()
            for d in range(len(docMask)):
                start = self.doc_range[docMask[d]]
                stop = self.doc_range[docMask[d] + 1]
                newTrueZList.append(TrueZ[start:stop])
            newTrueZ = np.hstack(newTrueZList)
            assert newTrueZ.size == newDocRange[-1]
        else:
            newTrueZ = None
        return GroupXData(newX, newDocRange,
                          Xprev=newXprev,
                          nDocTotal=nDocTotal,
                          TrueZ=newTrueZ)

    def add_data(self, XDataObj):
        """ Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        self.Data grows by adding all units from provided DataObj.
        """
        if not self.dim == XDataObj.dim:
            raise ValueError("Dimensions must match!")
        self.nObs += XDataObj.nObs
        self.nDocTotal += XDataObj.nDocTotal
        self.nDoc += XDataObj.nDoc
        self.X = np.vstack([self.X, XDataObj.X])
        if hasattr(self, 'Xprev'):
            self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])

        new_doc_range = XDataObj.doc_range[1:] + self.doc_range[-1]
        self.doc_range = np.hstack([self.doc_range, new_doc_range])
        self._check_dims()

    def get_random_sample(self, nDoc, randstate=np.random):
        nDoc = np.minimum(nDoc, self.nDoc)
        mask = randstate.permutation(self.nDoc)[:nDoc]
        Data = self.select_subset_by_mask(mask, doTrackFullSize=False)
        return Data

    def __str__(self):
        return self.X.__str__()

    def getRawDataAsSharedMemDict(self):
        ''' Create dict with copies of raw data as shared memory arrays
        '''
        dataShMemDict = dict()
        dataShMemDict['X'] = numpyToSharedMemArray(self.X)
        dataShMemDict['doc_range'] = numpyToSharedMemArray(self.doc_range)
        dataShMemDict['nDocTotal'] = self.nDocTotal
        if hasattr(self, 'Xprev'):
            dataShMemDict['Xprev'] = numpyToSharedMemArray(self.Xprev)
        return dataShMemDict

    def getDataSliceFunctionHandle(self):
        """ Return function handle that can make data slice objects.

        Useful with parallelized algorithms,
        when we need to use shared memory.

        Returns
        -------
        f : function handle
        """
        return makeDataSliceFromSharedMem


def makeDataSliceFromSharedMem(dataShMemDict,
                               cslice=(0, None),
                               batchID=None):
    """ Create data slice from provided raw arrays and slice indicators.

    Returns
    -------
    Dslice : namedtuple with same fields as GroupXData object
        * X
        * nObs
        * nObsTotal
        * dim
    Represents subset of documents identified by cslice tuple.

    Example
    -------
    >>> Data = GroupXData(np.random.rand(25,2), doc_range=[0,4,12,25])
    >>> shMemDict = Data.getRawDataAsSharedMemDict()
    >>> Dslice = makeDataSliceFromSharedMem(shMemDict)
    >>> np.allclose(Data.X, Dslice.X)
    True
    >>> np.allclose(Data.nObs, Dslice.nObs)
    True
    >>> Data.dim == Dslice.dim
    True
    >>> Aslice = makeDataSliceFromSharedMem(shMemDict, (0, 2))
    >>> Aslice.nDoc
    2
    >>> np.allclose(Aslice.doc_range, Dslice.doc_range[0:(2+1)])
    True
    """
    if batchID is not None and batchID in dataShMemDict:
        dataShMemDict = dataShMemDict[batchID]

    # Make local views (NOT copies) to shared mem arrays
    doc_range = sharedMemToNumpyArray(dataShMemDict['doc_range'])
    X = sharedMemToNumpyArray(dataShMemDict['X'])
    nDocTotal = int(dataShMemDict['nDocTotal'])

    dim = X.shape[1]
    if cslice is None:
        cslice = (0, doc_range.size - 1)
    elif cslice[1] is None:
        cslice = (0, doc_range.size - 1)
    tstart = doc_range[cslice[0]]
    tstop = doc_range[cslice[1]]

    keys = ['X', 'Xprev', 'doc_range', 'nDoc', 'nObs', 'dim', 'nDocTotal']

    if 'Xprev' in dataShMemDict:
        Xprev = sharedMemToNumpyArray(dataShMemDict['Xprev'])[tstart:tstop]
    else:
        Xprev = None

    Dslice = namedtuple("GroupXDataTuple", keys)(
        X=X[tstart:tstop],
        Xprev=Xprev,
        doc_range=doc_range[cslice[0]:cslice[1] + 1] - doc_range[cslice[0]],
        nDoc=cslice[1] - cslice[0],
        nObs=tstop - tstart,
        dim=dim,
        nDocTotal=nDocTotal,
    )
    return Dslice
