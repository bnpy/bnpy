'''
LawAdvice.py

Network of 71 lawyers making up a single US law firm. An edge is present from 
i->j if lawyer i reported that he/she went to lawyer j for "professional advice."  This dataset does *not* contain true labels.

Note that the full version of this dataset contains more information. For full 
dataset and more information, see: 
http://www.stats.ox.ac.uk/~snijders/siena/Lazega_lawyers_data.htm
'''



import numpy as np
import scipy.io
import os

from bnpy.data import GraphXData


# Get path to the .mat file with the data
datasetdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not os.path.isdir(datasetdir):
  raise ValueError('CANNOT FIND LAWADVICE DATASET DIRECTORY:\n' + datasetdir)

matfilepath = os.path.join(datasetdir, 'rawData', 'LawAdvice.mat')
if not os.path.isfile(matfilepath):
  raise ValueError('CANNOT FIND LAWADVICE DATASET MAT FILE:\n' + matfilepath)


########################################################### User-facing 
###########################################################
def get_data(**kwargs):
  Data = GraphXData.read_from_mat(matfilepath)
  Data.summary = get_data_info()
  Data.name = get_short_name()
  return Data

def get_data_info():
  return 'Network data showing the advice relationship between 71 lawyers from a single law firm.'

def get_short_name():
  return 'LawAdvice'



if __name__ == '__main__':
  import networkx as nx
  import matplotlib.pyplot as plt
  import bnpy.viz.RelationalViz as relviz

  Data = get_data()
  
  adjMtx = scipy.io.loadmat(matfilepath)
  A = adjMtx['X'].reshape((71,71))
  adjMtx = adjMtx['X'].reshape((71,71), order='C')
  G = nx.DiGraph(adjMtx)
  nx.draw_spring(G)
