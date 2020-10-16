'''
The init module gathers routines for initialization
'''

from bnpy.init import FromSaved
from bnpy.init import FromTruth
from bnpy.init import FromLP
from bnpy.init import FromScratchRelational
from bnpy.init import FromScratchGauss
from bnpy.init import FromScratchMult
from bnpy.init import FromScratchBern
from bnpy.init import FromScratchBregman
from bnpy.init import FromScratchBregmanMixture

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
