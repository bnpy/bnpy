"""
Classes
-----
GraphXData
    Data object for holding dense observations about edges of a network/graph.
    Organized as a list of edges, each with associated observations in field X.
"""
from builtins import *
import numpy as np
import scipy.io

from scipy.sparse import csc_matrix

from bnpy.util import as1D, as2D, as3D, toCArray
from .XData import XData

class GraphXData(XData):

    ''' Dataset object for dense observations about edges in network/graph.

    Attributes
    -------
    edges : 2D array, shape nEdges x 2
        Each row gives the source and destination node of an observed edge.
    X : 2D array, shape nEdges x D
        Row e contains the vector observation associated with edge e.
    nEdges : int
        Number of edges in current, in-memory batch.
        Always equal to edges.shape[0].
    nEdgesTotal : int
        Number of edges in whole dataset, across all batches.
        Always >= nEdges.
    nNodesTotal : int
        Total number of nodes in the dataset.
    nodeNames : (optional) list of size nNodes
        Human-readable names for each node.
    nodeZ : (optional) 1D array, size nNodes
        int cluster assignment for each node in "ground-truth" labeling.

    Optional Attributes
    -------------------
    TrueParams : dict
        Holds dataset's true parameters, including fields
        * Z :
        * w : 2D array, size K x K
            w[j,k] gives probability of edge between block j and block k

    Example
    --------
    >>> import numpy as np
    >>> from bnpy.data import GraphXData
    >>> AdjMat = np.asarray([[0, 1, 1], \
                             [0, 0, 1], \
                             [1, 0, 0]])
    >>> Data = GraphXData(AdjMat=AdjMat)
    >>> Data.nNodesTotal
    3
    >>> Data.nodes
    array([0, 1, 2])
    '''

    def __init__(self, edges=None, X=None,
                 AdjMat=None,
                 nNodesTotal=None, nEdgesTotal=None,
                 nNodes=None,
                 TrueParams=None,
                 nodeNames=None, nodeZ=None,
                 **kwargs):
        ''' Construct a GraphXData object.

        Pass either a full adjacency matrix (nNodes x nNodes x D),
        or a list of edges and associated observations.

        Args
        -----
        edges : 2D array, shape nEdges x 2
        X : 2D array, shape nEdges x D
        AdjMat : 3D array, shape nNodes x nNodes x D
            Defines adjacency matrix of desired graph.
            Assumes D=1 if 2D array specified.

        Returns
        --------
        Data : GraphXData
        '''
        self.isSparse = False
        self.TrueParams = TrueParams

        if AdjMat is not None:
            AdjMat = np.asarray(AdjMat)
            if AdjMat.ndim == 2:
                AdjMat = AdjMat[:, :, np.newaxis]
            nNodes = AdjMat.shape[0]
            edges = makeEdgesForDenseGraphWithNNodes(nNodes)
            X = np.zeros((edges.shape[0], AdjMat.shape[-1]))
            for eid, (i,j) in enumerate(edges):
                X[eid] = AdjMat[i,j]

        if AdjMat is None and (X is None or edges is None):
            raise ValueError(
                'Must specify adjacency matrix AdjMat, or ' +
                'a list of edges and corresponding dense observations X')

        # Create core attributes
        self.edges = toCArray(as2D(edges), dtype=np.int32)
        self.X = toCArray(as2D(X), dtype=np.float64)

        # Verify all edges are unique (raise error otherwise)
        N = self.edges.max() + 1
        edgeAsBaseNInteger = self.edges[:,0]*N + self.edges[:,1]
        nUniqueEdges = np.unique(edgeAsBaseNInteger).size
        if nUniqueEdges < self.edges.shape[0]:
            raise ValueError("Provided edges must be unique.")

        # Discard self loops
        nonselfloopmask = self.edges[:,0] != self.edges[:,1]
        if np.sum(nonselfloopmask) < self.edges.shape[0]:
            self.edges = self.edges[nonselfloopmask].copy()
            self.X = self.X[nonselfloopmask].copy()

        self._set_size_attributes(nNodesTotal=nNodesTotal,
                                  nEdgesTotal=nEdgesTotal)
        self._verify_attributes()

        if TrueParams is None:
            if nodeZ is not None:
                self.TrueParams = dict()
                self.TrueParams['nodeZ'] = nodeZ
        else:
            self.TrueParams = TrueParams
        if nodeNames is not None:
            self.nodeNames = nodeNames
        # TODO Held out data

    def _verify_attributes(self):
        ''' Basic runtime checks to make sure attribute dims are correct.
        '''
        assert self.edges.ndim == 2
        assert self.edges.min() >= 0
        assert self.edges.max() < self.nNodes
        nSelfLoops = np.sum(self.edges[:,0] == self.edges[:,1])
        assert nSelfLoops == 0
        assert self.nEdges <= self.nEdgesTotal

    def _set_size_attributes(self, nNodesTotal=None, nEdgesTotal=None):
        ''' Set internal fields that define sizes/dims.

        Post condition
        --------------
        Fields nNodes and nNodes total have proper, consistent values.
        '''
        if nEdgesTotal is None:
            self.nEdgesTotal = self.edges.shape[0]
        else:
            self.nEdgesTotal = nEdgesTotal
        if nNodesTotal is None:
            self.nNodesTotal = self.edges.max() + 1
        else:
            self.nNodesTotal = nNodesTotal

        self.nNodes = self.nNodesTotal
        self.nEdges = self.edges.shape[0]
        self.nObs = self.nEdges
        self.dim = self.X.shape[1]


    def get_stats_summary(self):
        ''' Returns human-readable summary of this dataset's basic properties
        '''
        s = 'Graph with %d nodes, %d edges and %d-dimensional observations' % (
            self.nNodesTotal, self.nEdges, self.dim)
        return s

    def get_total_size(self):
        return self.nEdgesTotal

    def get_size(self):
        return self.nEdges

    def select_subset_by_mask(self, mask, doTrackFullSize=True, **kwargs):
        ''' Creates new GraphXData object using a subset of edges.

        Args
        ----
        mask : 1D array_like
            Contains integer ids of edges to keep.
        doTrackFullSize : boolean
            if True, retain nObsTotal and nNodesTotal attributes.

        Returns
        -------
        subsetData : GraphXData object
        '''
        mask = np.asarray(mask, dtype=np.int32)
        edges = self.edges[mask]
        X = self.X[mask]

        if doTrackFullSize:
            nEdgesTotal = self.nEdgesTotal
            nNodesTotal = self.nNodesTotal
        else:
            nEdgesTotal = None
            nNodesTotal = None
        return GraphXData(edges=edges, X=X,
                          nNodesTotal=nNodesTotal,
                          nEdgesTotal=nEdgesTotal,
                          )

    def add_data(self, otherDataObj):
        ''' Updates (in-place) this object by adding new nodes.
        '''
        self.X = np.vstack([self.X, otherDataObj.X])
        self.edges = np.vstack([self.edges, otherDataObj.edges])
        self._set_size_attributes(
            nNodesTotal=self.nNodesTotal+otherDataObj.nNodesTotal,
            nEdgesTotal=self.nEdgesTotal+otherDataObj.nEdgesTotal)

    def toAdjacencyMatrix(self):
        ''' Return adjacency matrix representation of this dataset.

        Returns
        -------
        AdjMat : 3D array, nNodes x nNodes x D
        '''
        AdjMat = np.empty((self.nNodes, self.nNodes, self.dim))
        AdjMat[:] = np.nan
        AdjMat[self.edges[:,0], self.edges[:,1]] = self.X
        return AdjMat

    def getSparseSrcNodeMat(self):
        ''' Get sparse indicator matrix to sum edges by source node.

        Returns
        -------
        S : 2D sparse matrix, size nNode x nEdges
            S[s,e] is 1 iff node s is the source for edge e
            Take any vector or matrix A with first axis for edges, and
            S * A will sum over rows of A by source node.
        '''
        try:
            return self.SrcNodeMat
        except AttributeError:
            self.SrcNodeMat = csc_matrix(
                (np.ones(self.nEdges),
                 self.edges[:,0],
                 np.arange(self.nEdges+1)),
                shape=(self.nNodes, self.nEdges))
        return self.SrcNodeMat

    def getSparseRcvNodeMat(self):
        ''' Get sparse indicator matrix to sum edges by receiver node.

        Returns
        -------
        S : 2D sparse matrix, size nNode x nEdges
            S[s,e] is 1 iff node s is the receiver for edge e
            Take any vector or matrix A with first axis for edges, and
            S * A will sum over rows of A by receiver node.
        '''
        try:
            return self.RcvNodeMat
        except AttributeError:
            self.RcvNodeMat = csc_matrix(
                (np.ones(self.nEdges),
                 self.edges[:,1],
                 np.arange(self.nEdges+1)),
                shape=(self.nNodes, self.nEdges))
        return self.RcvNodeMat

    @classmethod
    def LoadFromFile(cls, filepath, **kwargs):
        ''' Static constructor for loading data from disk into XData instance
        '''
        if filepath.endswith('.mat'):
            return cls.read_from_mat(filepath, **kwargs)
        elif filepath.endswith('.txt'):
            return cls.read_from_txt(filepath, **kwargs)
        raise NotImplemented('File extension not supported')

    @classmethod
    def read_from_graphtxtfile(cls, filepath,
                      nEdgesTotal=None, nNodesTotal=None,
                      settingspath=None,
                      **kwargs):
        ''' Static constructor loading .graph file into GraphXData instance.
        '''
        if settingspath is not None:
            with open(settingspath, 'r') as f:
                for line in f.readlines():
                    if line.count('='):
                        fields = [f.strip() for f in line.split('=')]
                        assert len(fields) == 2
                        if fields[0] == 'N' or fields[0] == 'nNodesTotal':
                            nNodesTotal = int(fields[1])
                        if fields[0] == 'E' or fields[0] == 'nEdgesTotal':
                            nEdgesTotal = int(fields[1])
        txtData = np.loadtxt(filepath, dtype=np.int32)
        assert txtData.ndim == 2
        assert txtData.shape[1] == 4
        edges = txtData[:, [1,2]]
        X = as2D(txtData[:, 3])
        # Make sure X and edges have correct dims
        if X.shape[0] != edges.shape[0]:
            X = X.T
        return cls(nNodesTotal=nNodesTotal, nEdgesTotal=nEdgesTotal,
                   edges=edges, X=X)
    @classmethod
    def read_from_txt(cls, filepath,
                      nEdgesTotal=None, nNodesTotal=None, **kwargs):
        ''' Static constructor loading .txt file into GraphXData instance.
        '''
        txt = np.loadtxt(filepath, dtype=np.float64)
        edges = txt[:, :2]
        X = txt[:, 2:]
        return cls(nNodesTotal=nNodesTotal, nEdgesTotal=nEdgesTotal,
                   edges=edges, X=X)

    @classmethod
    def read_from_mat(cls, matfilepath, **kwargs):
        ''' Static constructor loading .mat file into GraphXData instance.

        Keyword args (optional)
        ---------------
        nNodesTotal : int
        nEdgesTotal : int
        '''
        InDict = scipy.io.loadmat(matfilepath, **kwargs)
        if 'TrueZ' in InDict:
            InDict['TrueParams'] = {'Z': InDict['TrueZ']}
        InDict.update(kwargs) # nNodesTotal, nEdgesTotal passed in here
        return cls(**InDict)


def makeEdgesForDenseGraphWithNNodes(N):
    ''' Make edges array for a directed graph with N nodes.

    Returns
    --------
    edges : 2D array, shape nEdges x 2
        contains all non-self-loop edges
    '''
    edges = list()
    for s in range(N):
        for t in range(N):
            if s == t:
                continue
            edges.append((s,t))
    return np.asarray(edges, dtype=np.int32)
