''' bnpy module __init__ file
'''
from builtins import *
import os
import sys
import psutil

# Configure PYTHONPATH before importing any bnpy modules
ROOT_PATH = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-2])
# sys.path.append(os.path.join(BNPYROOTDIR, 'datasets/'))
# sys.path.append(os.path.join(BNPYROOTDIR, 'third-party/'))
# sys.path.append(os.path.join(ROOT_PATH, 'third-party/anchorwordtopics/'))

DATASET_PATH = os.path.join(ROOT_PATH, 'bnpy/datasets/')

from bnpy import data
from bnpy import suffstats
from bnpy import util

from bnpy import allocmodel
from bnpy import obsmodel
from bnpy.HModel import HModel

from bnpy import ioutil
from bnpy import init
from bnpy import learnalg
from bnpy import birthmove
from bnpy import mergemove
from bnpy import deletemove

from bnpy import callbacks

from bnpy import Run

# Convenient aliases to existing functions
run = Run.run
load_model_at_lap = ioutil.ModelReader.load_model_at_lap
save_model = ioutil.ModelWriter.save_model
make_initialized_model = Run.make_initialized_model


__all__ = ['run', 'Run', 'learnalg', 'allocmodel', 'obsmodel', 'suffstats',
           'HModel', 'init', 'util', 'ioutil']

# Optional viz package for plotting
try:
    from matplotlib import pylab
    from bnpy import viz
    __all__.append('viz')
except ImportError:
    print("Error importing matplotlib. Plotting disabled.")
    print("Fix by making sure this produces a figure window on your system")
    print(" >>> from matplotlib import pylab; pylab.figure(); pylab.show();")
