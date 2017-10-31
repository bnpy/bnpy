''' bnpy module __init__ file
'''
from builtins import *
import os
import sys
import psutil

def getCurMemCost_MiB():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem_MiB = process.memory_info_ex().rss / float(2 ** 20)
    return mem_MiB

def pprintCurMemCost(label=''):
    # return the memory usage in MB
    mem_MiB = getCurMemCost
    print("%.3f MiB | %s" % (mem_MiB, label))

# Configure PYTHONPATH before importing any bnpy modules
ROOT_PATH = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-2])
# sys.path.append(os.path.join(BNPYROOTDIR, 'datasets/'))
# sys.path.append(os.path.join(BNPYROOTDIR, 'third-party/'))
# sys.path.append(os.path.join(ROOT_PATH, 'third-party/anchorwordtopics/'))

DATASET_PATH = os.path.join(ROOT_PATH, 'bnpy/datasets/')

from . import data
from . import suffstats
from . import util

from . import allocmodel
from . import obsmodel
from .HModel import HModel

from . import ioutil
from . import init
from . import learnalg
from . import birthmove
from . import mergemove
from . import deletemove

from . import callbacks

from . import Run

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
    from . import viz
    __all__.append('viz')
except ImportError:
    print("Error importing matplotlib. Plotting disabled.")
    print("Fix by making sure this produces a figure window on your system")
    print(" >>> from matplotlib import pylab; pylab.figure(); pylab.show();")
