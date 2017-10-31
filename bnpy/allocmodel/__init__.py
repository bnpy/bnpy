from .AllocModel import AllocModel

from .mix.FiniteMixtureModel import FiniteMixtureModel
from .mix.DPMixtureModel import DPMixtureModel
from .mix.DPMixtureRestrictedLocalStep import make_xPiVec_and_emptyPi

from .topics.FiniteTopicModel import FiniteTopicModel
from .topics.HDPTopicModel import HDPTopicModel

from .hmm.FiniteHMM import FiniteHMM
from .hmm.HDPHMM import HDPHMM

from .relational.FiniteSMSB import FiniteSMSB
from .relational.FiniteMMSB import FiniteMMSB
from .relational.FiniteAssortativeMMSB import FiniteAssortativeMMSB
from .relational.HDPMMSB import HDPMMSB
from .relational.HDPAssortativeMMSB import HDPAssortativeMMSB


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
