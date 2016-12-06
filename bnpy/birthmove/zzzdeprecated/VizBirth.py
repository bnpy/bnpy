import numpy as np


def viz_birth_proposal(curModel, propModel, Plan, **kwargs):
    if str(type(curModel.obsModel)).count('Gauss') > 0:
        _viz_Gauss(curModel, propModel, Plan, **kwargs)
    else:
        _viz_Mult(curModel, propModel, Plan, **kwargs)


def _viz_Gauss(curModel, propModel, Plan,
               curELBO=None, propELBO=None, block=False, **kwargs):
    from ..viz import GaussViz
    from matplotlib import pylab
    pylab.figure()
    h = pylab.subplot(1, 2, 1)
    GaussViz.plotGauss2DFromHModel(curModel, compsToHighlight=Plan['ktarget'])
    h = pylab.subplot(1, 2, 2)
    newCompIDs = np.arange(curModel.obsModel.K, propModel.obsModel.K)
    GaussViz.plotGauss2DFromHModel(propModel, compsToHighlight=newCompIDs)
    pylab.show(block=block)


def _viz_Mult(curModel, propModel, Plan,
              curELBO=None, propELBO=None, block=False, **kwargs):
    from ..viz import BarsViz
    from matplotlib import pylab
    BarsViz.plotBarsFromHModel(curModel, compsToHighlight=Plan['ktarget'])
    BarsViz.plotBarsFromHModel(propModel, compsToHighlight=None)
    if curELBO is not None:
        pylab.xlabel("%.3e" % (propELBO - curELBO))
    pylab.show(block=block)
