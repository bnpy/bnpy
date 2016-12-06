'''
The init module gathers routines for initialization
'''

import FromSaved
import FromTruth
import FromLP
import FromScratchRelational
import FromScratchGauss
import FromScratchMult
import FromScratchBern
import FromScratchBregman
import FromScratchBregmanMixture

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
