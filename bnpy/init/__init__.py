'''
The init module gathers routines for initialization
'''

from . import FromSaved
from . import FromTruth
from . import FromLP
from . import FromScratchRelational
from . import FromScratchGauss
from . import FromScratchMult
from . import FromScratchBern
from . import FromScratchBregman
from . import FromScratchBregmanMixture

# from FromScratchMult import initSSByBregDiv_Mult
# from FromScratchBern import initSSByBregDiv_Bern
# from FromScratchGauss import initSSByBregDiv_Gauss
# from FromScratchGauss import initSSByBregDiv_ZeroMeanGauss

def initSSByBregDiv(curModel=None, **kwargs):
	obsName = curModel.getObsModelName()
	if obsName.count('Mult'):
		return initSSByBregDiv_Mult(curModel=curModel, **kwargs)
	elif obsName.count('ZeroMeanGauss'):
		return initSSByBregDiv_ZeroMeanGauss(curModel=curModel, **kwargs)
	elif obsName.count('Gauss'):
		return initSSByBregDiv_Gauss(curModel=curModel, **kwargs)
	else:
		raise NotImplementedError("Unknown obsmodel name: " + obsName)
