from .DiagGaussObsModel import DiagGaussObsModel
from .GaussObsModel import GaussObsModel
from .ZeroMeanGaussObsModel import ZeroMeanGaussObsModel
from .AutoRegGaussObsModel import AutoRegGaussObsModel
from .MultObsModel import MultObsModel
from .BernObsModel import BernObsModel
from .GaussRegressYFromFixedXObsModel \
	import GaussRegressYFromFixedXObsModel
from .GaussRegressYFromDiagGaussXObsModel \
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
