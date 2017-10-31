from builtins import *
from matplotlib import pylab
import matplotlib.axis, matplotlib.scale
from matplotlib.ticker import \
    MaxNLocator, ScalarFormatter, NullLocator, NullFormatter

ExportInfo = dict(
    doExport=False,
    dpi=100,
    W_in=4,
    H_in=4,
)

def ConfigPylabDefaults(pylab, **kwargs):
    rcParams = pylab.rcParams
    rcParams['pdf.fonttype'] = 42  # Make fonts export as text (not bitmap)
    rcParams['ps.fonttype'] = 42  # Make fonts export as text (not bitmap)
    rcParams['text.usetex'] = False
    rcParams['legend.fontsize'] = 16
    rcParams['axes.titlesize'] = 18
    rcParams['axes.labelsize'] = 18
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    rcParams['figure.figsize'] = ExportInfo['W_in'], ExportInfo['H_in']
    rcParams['figure.dpi'] = ExportInfo['dpi']
    rcParams['figure.subplot.left'] = 0.15
    rcParams['figure.subplot.right'] = 0.95
    rcParams['figure.subplot.bottom'] = 0.15
    rcParams['figure.subplot.top'] = 0.95
    rcParams['savefig.dpi'] = ExportInfo['dpi']
    rcParams.update(kwargs)

def set_my_locators_and_formatters(self, axis):
    # choose the default locator and additional parameters
    if isinstance(axis, matplotlib.axis.XAxis):
        axis.set_major_locator(MaxNLocator(prune='lower'))
    elif isinstance(axis, matplotlib.axis.YAxis):
        axis.set_major_locator(MaxNLocator())
    # copy & paste from the original method
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())

# Set defaults for nbins in xticks and yticks
MaxNLocator.default_params['nbins']=5
matplotlib.scale.LinearScale.set_default_locators_and_formatters = \
    set_my_locators_and_formatters
