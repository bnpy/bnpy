'''
Classes
-------
XData : dataset_object
    Holds a 2D array X of exchangable observations
    Each observation is a dense row vector inside the array X
'''

import numpy as np
import scipy.io
import inspect
import os
from collections import namedtuple
import pandas as pd

from DataObj import DataObj
from bnpy.util import as1D, as2D, toCArray
from bnpy.util import numpyToSharedMemArray, sharedMemToNumpyArray


class XData(DataObj):

    """ Dataset object for dense vectors of real-valued observations.

    Attributes
    ------
    X : 2D array, size N x D
        each row is a single dense observation vector
    Xprev : 2D array, size N x D, optional
        "previous" observations for auto-regressive likelihoods
    Y : 1D array, size N, optional
        response or dependent variable for regression likelihoods
    n_examples : int
        number of in-memory observations for this instance
    nObsTotal : int
        total size of the dataset which in-memory X is a part of.
    n_dims : int
        number of dimensions
    dtype : type, default = 'auto'
        the type of each observation
    name : str
        String name of this dataset.
        Default: none
    TrueParams : dict
        key/value pairs represent names and arrays of true parameters

    Example
    -------
    >>> X = np.zeros((1000, 3)) # Create 1000x3 matrix
    >>> myData = XData(X) # Convert to an XData object
    >>> print myData.nObs
    1000
    >>> print myData.dim
    3
    >>> print myData.X.shape
    (1000, 3)
    >>> mySubset = myData.make_subset([0])
    >>> mySubset.X.shape
    (1, 3)
    >>> mySubset.X[0]
    array([ 0.,  0.,  0.])
    """

    @classmethod
    def LoadFromFile(cls, filepath, nObsTotal=None, **kwargs):
        ''' Constructor for loading data from disk into XData instance.
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, nObsTotal, **kwargs)
        try:
            X = np.load(filepath)
        except Exception as e:
            X = np.loadtxt(filepath)
        return cls(X, nObsTotal=nObsTotal, **kwargs)
 
    @classmethod
    def read_file(cls, filepath, **kwargs):
        ''' Constructor for loading data from disk into XData instance.
        '''
        if filepath.endswith('.npz'):
            return cls.read_npz(filepath, **kwargs)
        elif filepath.endswith('.mat'):
            return cls.read_mat(filepath, **kwargs)
        elif filepath.endswith('.csv'):
            return cls.read_csv(filepath, **kwargs)
        raise ValueError("Unrecognized file format: " + filepath)

    @classmethod
    def read_mat(
            cls, matfilepath, nObsTotal=None,
            variable_names=None, **kwargs):
        ''' Constructor for loading .mat file into XData instance.

        Returns
        -------
        dataset : XData object

        Examples
        --------
        >>> dataset_path = os.environ["BNPYDATADIR"]
        >>> dataset = XData.read_mat(
        ...     os.path.join(dataset_path, 'AsteriskK8', 'x_dataset.mat'))
        >>> dataset.dim
        2
        '''
        names, varargs, varkw, defaults = inspect.getargspec(scipy.io.loadmat)
        loadmatKwargs = dict()
        for key in kwargs:
            if key in names:
                loadmatKwargs[key] = kwargs[key]
        InDict = scipy.io.loadmat(matfilepath, **loadmatKwargs)
        if 'X' not in InDict:
            raise KeyError(
                'Stored matfile needs to have data in field named X')
        if nObsTotal is not None:
            InDict['nObsTotal'] = nObsTotal
        # Magically call __init__
        return cls(**InDict)

    @classmethod
    def read_npz(
            cls, npzfilepath, nObsTotal=None, **kwargs):
        ''' Constructor for loading .npz file into XData instance.

        Returns
        -------
        dataset : XData object
            
        Examples
        --------
        >>> dataset_path = os.environ["BNPYDATADIR"]
        >>> dataset = XData.read_npz(
        ...     os.path.join(dataset_path, 'AsteriskK8', 'x_dataset.npz'))
        >>> dataset.dim
        2
        '''
        npz_dict = dict(**np.load(npzfilepath))
        if 'X' not in npz_dict:
            raise KeyError(
                '.npz file needs to have data in field named X')
        if nObsTotal is not None:
            npz_dict['nObsTotal'] = nObsTotal
        # Magically call __init__
        return cls(**npz_dict)

    @classmethod
    def read_csv(
            cls, csvfilepath, nObsTotal=None, **kwargs):
        ''' Constructor for loading .csv file into XData instance.

        Returns
        -------
        dataset : XData object
            
        Examples
        --------
        >>> dataset_path = os.environ["BNPYDATADIR"]
        >>> dataset = XData.read_csv(
        ...     os.path.join(dataset_path, 'AsteriskK8', 'x_dataset.csv'))
        >>> dataset.dim
        2
        >>> dataset.column_names
        ['x_0', 'x_1']
        '''
        x_df = pd.read_csv(csvfilepath)
        return cls.from_dataframe(x_df)

    @classmethod
    def from_dataframe(
            cls, x_df, **kwargs):
        ''' Convert pandas dataframe into XData dataset object

        Returns
        -------
        dataset : XData object

        Examples
        --------
        >>> x_df = pd.DataFrame(np.zeros((3,2)), columns=['a', 'b'])
        >>> dataset = XData.from_dataframe(x_df)
        >>> dataset.dim
        2
        >>> dataset.nObs
        3
        >>> dataset.column_names
        ['a', 'b']
        '''
        # TODO row names
        all_column_names = x_df.columns
        y_column_names = [
            name for name in all_column_names if name.lower().startswith("y_")]
        z_column_names = [
            name for name in all_column_names if name.lower().startswith("z_")]
        x_column_names = [
            name for name in all_column_names
                if (name not in y_column_names)
                    and (name not in z_column_names)]
        X = np.asarray(x_df[x_column_names])
        if len(z_column_names) > 0:
            TrueZ = np.asarray(x_df[z_column_names[0]])
        else:
            TrueZ = None
        return cls(
            X=X,
            column_names=x_column_names,
            TrueZ=TrueZ,
            **kwargs)

    def __init__(self,
            X=None,
            nObsTotal=None,
            TrueZ=None,
            Xprev=None,
            Y=None,
            TrueParams=None,
            name=None,
            summary=None,
            dtype='auto',
            row_names=None,
            column_names=None,
            y_column_names=None,
            xprev_column_names=None,
            do_copy=True,
            **kwargs):
        ''' Constructor for XData instance given in-memory dense array X.

        Post Condition
        ---------
        self.X : 2D array, size N x D
            with standardized dtype, alignment, byteorder.
        '''
        if dtype == 'auto':
            dtype = X.dtype
        if not do_copy and X.dtype == dtype:
            self.X = as2D(X)
        else:
            self.X = as2D(toCArray(X, dtype=dtype))

        if Xprev is not None:
            self.Xprev = as2D(toCArray(Xprev, dtype=dtype))
        if Y is not None:
            self.Y = as2D(toCArray(Y, dtype=dtype))

        # Verify attributes are consistent
        self._set_dependent_params(nObsTotal=nObsTotal)
        self._check_dims(do_copy=do_copy)

        # Add optional true parameters / true hard labels
        if TrueParams is not None:
            self.TrueParams = TrueParams
        if TrueZ is not None:
            if not hasattr(self, 'TrueParams'):
                self.TrueParams = dict()
            self.TrueParams['Z'] = as1D(toCArray(TrueZ))
            self.TrueParams['K'] = np.unique(self.TrueParams['Z']).size
        if summary is not None:
            self.summary = summary
        if name is not None:
            self.name = str(name)

        # Add optional row names
        if row_names is None:
            self.row_names = map(str, range(self.nObs))
        else:
            assert len(row_names) == self.nObs
            self.row_names = map(str, row_names)

        # Add optional column names
        if column_names is None:
            self.column_names = map(lambda n: "dim_%d" % n, range(self.dim))
        else:
            assert len(column_names) == self.dim
            self.column_names = map(str, column_names)

    def _set_dependent_params(self, nObsTotal=None):
        self.nObs = self.X.shape[0]
        self.dim = self.X.shape[1]
        if nObsTotal is None:
            self.nObsTotal = self.nObs
        else:
            self.nObsTotal = nObsTotal

    def _check_dims(self, do_copy=False):
        assert self.X.ndim == 2
        if do_copy:
            assert self.X.flags.c_contiguous
            assert self.X.flags.owndata
            assert self.X.flags.aligned
            assert self.X.flags.writeable
        if hasattr(self, 'Y'):
            assert self.Y.shape[0] == self.X.shape[0]
        if hasattr(self, 'Xprev'):
            assert self.Xprev.shape[0] == self.X.shape[0]
            
    def get_size(self):
        """ Get number of observations in memory for this object.

        Returns
        ------
        n : int
        """
        return self.nObs

    def get_total_size(self):
        """ Get total number of observations for this dataset.

        This may be much larger than self.nObs.

        Returns
        ------
        n : int
        """
        return self.nObsTotal

    def get_dim(self):
        return self.dim

    def get_example_names(self):
        if hasattr(self, 'row_names'):
            return self.row_names
        else:
            return map(str, np.arange(self.nObs))

    def get_text_summary(self):
        ''' Get human-readable description of this dataset.

        Returns
        -------
        s : string
        '''
        if hasattr(self, 'summary'):
            s = self.summary
        else:
            s = 'X Data'
        return s

    def get_stats_summary(self):
        ''' Get human-readable summary of this dataset's basic properties

        Returns
        -------
        s : string
        '''
        s = '  num examples: %d\n' % (self.nObs)
        s += '  num dims: %d' % (self.dim)
        return s

    def make_subset(
            self,
            example_id_list=None,
            doTrackFullSize=True,
            doTrackTruth=False):
        ''' Get subset of this dataset identified by provided unit IDs.

        Parameters
        -------
        keep_id_list : 1D array_like
            Identifies units (rows) of X to use for subset.
        doTrackFullSize : boolean
            If True, return DataObj with same nObsTotal value as this
            dataset. If False, returned DataObj has smaller size.

        Returns
        -------
        Dchunk : bnpy.data.XData instance
        '''
        if hasattr(self, 'Xprev'):
            newXprev = self.Xprev[example_id_list]
        else:
            newXprev = None
        if hasattr(self, 'Y'):
            newY = self.Y[example_id_list]
        else:
            newY = None
        newX = self.X[example_id_list]

        if hasattr(self, 'alwaysTrackTruth'):
            doTrackTruth = doTrackTruth or self.alwaysTrackTruth
        hasTrueZ = hasattr(self, 'TrueParams') and 'Z' in self.TrueParams
        if doTrackTruth and hasTrueZ:
            TrueZ = self.TrueParams['Z']
            newTrueZ = TrueZ[example_id_list]
        else:
            newTrueZ = None

        if doTrackFullSize:
            nObsTotal = self.nObsTotal
        else:
            nObsTotal = None

        return XData(
            X=newX,
            Xprev=newXprev,
            Y=newY,
            nObsTotal=nObsTotal,
            row_names=[self.row_names[i] for i in example_id_list],
            TrueZ=newTrueZ,
            )

    def add_data(self, XDataObj):
        """ Appends (in-place) provided dataset to this dataset.

        Post Condition
        -------
        self.Data grows by adding all units from provided DataObj.
        """
        if not self.dim == XDataObj.dim:
            raise ValueError("Dimensions must match!")
        self.nObs += XDataObj.nObs
        self.nObsTotal += XDataObj.nObsTotal
        self.X = np.vstack([self.X, XDataObj.X])
        if hasattr(self, 'Xprev'):
            assert hasattr(XDataObj, 'Xprev')
            self.Xprev = np.vstack([self.Xprev, XDataObj.Xprev])
        if hasattr(self, 'Y'):
            assert hasattr(XDataObj, 'Y')
            self.Y = np.vstack([self.Y, XDataObj.Y])
        self._check_dims()

    def get_random_sample(self, n_examples, randstate=np.random):
        n_examples = np.minimum(n_examples, self.nObs)
        mask = randstate.permutation(self.nObs)[:n_examples]
        Data = self.make_subset(mask, doTrackFullSize=False)
        return Data

    def __str__(self):
        return self.X.__str__()

    def getRawDataAsSharedMemDict(self):
        ''' Create dict with copies of raw data as shared memory arrays
        '''
        dataShMemDict = dict()
        dataShMemDict['X'] = numpyToSharedMemArray(self.X)
        dataShMemDict['nObsTotal'] = self.nObsTotal
        if hasattr(self, 'Xprev'):
            dataShMemDict['Xprev'] = numpyToSharedMemArray(self.Xprev)
        if hasattr(self, 'Y'):
            dataShMemDict['Y'] = numpyToSharedMemArray(self.Y)
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

    def to_csv(self, csv_file_path, **kwargs):
        ''' Convert this dataset object to a comma-separated value file.

        Post Condition
        --------------
        CSV file written to disk.

        Examples
        --------
        >>> dataset = XData(X=np.zeros((3,2)), column_names=['a', 'b'])
        >>> dataset.to_csv('/tmp/x_dataset.csv')
        '''
        X_df = self.to_dataframe()
        X_df.to_csv(csv_file_path)

    def to_dataframe(self):
        ''' Convert this dataset object to a dictionary.

        Returns
        -------
        x_dict : dict with key for each attribute

        Examples
        --------
        >>> dataset = XData(X=np.zeros((3,2)), column_names=['a', 'b'])
        >>> x_df = dataset.to_dataframe()
        >>> print x_df
             a    b
        0  0.0  0.0
        1  0.0  0.0
        2  0.0  0.0
        '''
        x_df = pd.DataFrame(
            data=self.X,
            index=self.get_example_names(),
            columns=self.column_names)
        return x_df

    def to_dict(self, **kwargs):
        ''' Convert this dataset object to a dictionary.

        Returns
        -------
        x_dict : dict with key for each attribute

        Examples
        --------
        >>> dataset = XData(np.zeros((5,3)))
        >>> my_dict = dataset.to_dict()
        >>> "X" in my_dict
        True
        '''
        return self.__dict__

def makeDataSliceFromSharedMem(dataShMemDict,
                               cslice=(0, None),
                               batchID=None):
    """ Create data slice from provided raw arrays and slice indicators.

    Returns
    -------
    Dslice : namedtuple with same fields as XData object
        * X
        * n_examples
        * nObsTotal
        * n_dims
    Represents subset of documents identified by cslice tuple.

    Example
    -------
    >>> dataset = XData(np.random.rand(25,2))
    >>> shMemDict = dataset.getRawDataAsSharedMemDict()
    >>> cur_slice = makeDataSliceFromSharedMem(shMemDict)
    >>> np.allclose(dataset.X, cur_slice.X)
    True
    >>> np.allclose(dataset.nObs, cur_slice.nObs)
    True
    >>> dataset.dim == cur_slice.dim
    True
    >>> a_slice = makeDataSliceFromSharedMem(shMemDict, (0, 2))
    >>> a_slice.nObs
    2
    """
    if batchID is not None and batchID in dataShMemDict:
        dataShMemDict = dataShMemDict[batchID]

    # Make local views (NOT copies) to shared mem arrays
    X = sharedMemToNumpyArray(dataShMemDict['X'])
    nObsTotal = int(dataShMemDict['nObsTotal'])

    N, dim = X.shape
    if cslice is None:
        cslice = (0, N)
    elif cslice[1] is None:
        cslice = (0, N)

    keys = ['X', 'Xprev', 'n_examples', 'n_dims', 'nObsTotal']

    if 'Xprev' in dataShMemDict:
        Xprev = sharedMemToNumpyArray(
            dataShMemDict['Xprev'])[cslice[0]:cslice[1]]
    else:
        Xprev = None

    if 'Y' in dataShMemDict:
        Y = sharedMemToNumpyArray(
            dataShMemDict['Y'])[cslice[0]:cslice[1]]
    else:
        Y = None

    return XData(
        X=X[cslice[0]:cslice[1]],
        Xprev=Xprev,
        Y=Y,
        n_examples=cslice[1] - cslice[0],
        nObsTotal=nObsTotal,
        do_copy=False)
