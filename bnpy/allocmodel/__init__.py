from bnpy.allocmodel.AllocModel import AllocModel

from bnpy.allocmodel.mix.FiniteMixtureModel import FiniteMixtureModel
from bnpy.allocmodel.mix.DPMixtureModel import DPMixtureModel
from bnpy.allocmodel.mix.DPMixtureRestrictedLocalStep import make_xPiVec_and_emptyPi

from bnpy.allocmodel.topics.FiniteTopicModel import FiniteTopicModel
from bnpy.allocmodel.topics.HDPTopicModel import HDPTopicModel

from bnpy.allocmodel.hmm.FiniteHMM import FiniteHMM
from bnpy.allocmodel.hmm.HDPHMM import HDPHMM

from bnpy.allocmodel.relational.FiniteSMSB import FiniteSMSB
from bnpy.allocmodel.relational.FiniteMMSB import FiniteMMSB
from bnpy.allocmodel.relational.FiniteAssortativeMMSB import FiniteAssortativeMMSB
from bnpy.allocmodel.relational.HDPMMSB import HDPMMSB
from bnpy.allocmodel.relational.HDPAssortativeMMSB import HDPAssortativeMMSB


AllocModelConstructorsByName = {
    'FiniteMixtureModel': FiniteMixtureModel,
    'DPMixtureModel': DPMixtureModel,
    'FiniteTopicModel': FiniteTopicModel,
    'HDPTopicModel': HDPTopicModel,
    'FiniteHMM': FiniteHMM,
    'HDPHMM': HDPHMM,
    'FiniteSMSB': FiniteSMSB,
    'FiniteMMSB': FiniteMMSB,
    'FiniteAssortativeMMSB': FiniteAssortativeMMSB,
    'HDPMMSB': HDPMMSB,
    'HDPAssortativeMMSB': HDPAssortativeMMSB,
}

AllocModelNameSet = set(AllocModelConstructorsByName.keys())

__all__ = ['AllocModel']
for name in AllocModelConstructorsByName:
    __all__.append(name)
