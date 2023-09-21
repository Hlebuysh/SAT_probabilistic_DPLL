from enum import Enum, auto


class RecType(Enum):
    FALSE = auto()
    TRUE = auto()
    BOTH = auto()


class HeuristicType(Enum):
    STANDART_DPLL = 'ST'
    MAXIMUM_LIKELIHOOD_ESTIMATION = 'MLE'
    MAXIMUM_POSTERIORI_ESTIMATION = 'MPE'
    MOST_CONSTRAINED_VARIABLE = 'MCV'
    JEROSLOW_WANG = 'JW'

class EvolutionType(Enum):
    LOCAL_SEARCH = 'LS'


class NextRec:
    def __init__(self, var, rec_type):
        self.var = var
        self.type = rec_type
