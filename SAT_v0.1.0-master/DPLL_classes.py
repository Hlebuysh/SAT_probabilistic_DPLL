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
    GRADIENT_DESCENT = 'GD'
    SIMULATED_ANNEALING = 'SA'
    INTEGER_LINEAR_PROGRAMMING = 'ILP'
    MIXED_INTEGER_LINEAR_PROGRAMMING = 'MILP'


class NextRec:
    def __init__(self, var, rec_type):
        self.var = var
        self.type = rec_type
