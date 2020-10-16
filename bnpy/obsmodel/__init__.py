
from bnpy.obsmodel.DiagGaussObsModel import DiagGaussObsModel
from bnpy.obsmodel.GaussObsModel import GaussObsModel
from bnpy.obsmodel.ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from bnpy.obsmodel.AutoRegGaussObsModel import AutoRegGaussObsModel
from bnpy.obsmodel.MultObsModel import MultObsModel
from bnpy.obsmodel.BernObsModel import BernObsModel
from bnpy.obsmodel.GaussRegressYFromFixedXObsModel \
	import GaussRegressYFromFixedXObsModel
from bnpy.obsmodel.GaussRegressYFromDiagGaussXObsModel \
	import GaussRegressYFromDiagGaussXObsModel

ObsModelConstructorsByName = {
    'DiagGauss': DiagGaussObsModel,
    'Gauss': GaussObsModel,
    'ZeroMeanGauss': ZeroMeanGaussObsModel,
    'AutoRegGauss': AutoRegGaussObsModel,
    'GaussRegressYFromFixedX': GaussRegressYFromFixedXObsModel,
    'GaussRegressYFromDiagGaussX': GaussRegressYFromDiagGaussXObsModel,
    'Mult': MultObsModel,
    'Bern': BernObsModel,
}

# Make constructor accessible by nickname and fullname
# Nickname = 'Gauss'
# Fullname = 'GaussObsModel'
for val in list(ObsModelConstructorsByName.values()):
    fullname = str(val.__name__)
    ObsModelConstructorsByName[fullname] = val

ObsModelNameSet = set(ObsModelConstructorsByName.keys())
