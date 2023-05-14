import sys
from enum import Enum, auto
from time import perf_counter


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


## RecType используется для определения типов записей (record types) в NextRec,
# чтобы указать, какие ветви должны быть продолжены в алгоритме DPLL.
## RecType определяет три возможных значения: FALSE, TRUE и BOTH, которые указывают,
# должны ли ветви быть продолжены, только если литерал является ложным, только если он истинен, или в обоих случаях.
## RecType не используется явно где-либо в коде. Однако он используется неявно в методе get_next_rec,
# который возвращает объект NextRec, содержащий литерал и его тип (одно из значений RecType).
## Позже этот объект используется в функции gen_branch, которая определяет,
# какие литералы должны быть добавлены к ветви, в зависимости от типа NextRec.

class NextRec:
    def __init__(self, var, rec_type):
        self.var = var
        self.type = rec_type


## С помощью класса NextRec мы можем получить следующую переменную,
# которая будет использоваться для дальнейшего выполнения алгоритма.


heuristic_type = 'ALL'
input_file = 'input.txt'
output_file = 'output.txt'

def simplify_cnf(clauses):
    simplified_clauses = [clause for clause in clauses if
                          not any(literal.lower() in clause for literal in clause if literal.isupper())]

    unit_clauses = [clause[0] for clause in simplified_clauses if len(clause) == 1 and clause[0].islower()]
    while unit_clauses:
        unit_literal = unit_clauses[0]
        simplified_clauses = [clause for clause in simplified_clauses if unit_literal not in clause]
        simplified_clauses = [[literal for literal in clause if literal != unit_literal.upper()] for clause in
                              simplified_clauses]
        unit_clauses = [clause[0] for clause in simplified_clauses if len(clause) == 1 and clause[0].islower()]

    return simplified_clauses


def get_next_rec(clause: list[list[str]], vars: list[str], probabilities: list[float]) -> NextRec:
    def maximumLikelihoodEstimationHeuristic() -> str:
        clause_vars = {literal.lower() for disjunction in clause for literal in disjunction}

        variable = vars[0]
        close_probability = probabilities[0]
        for i in range(1, len(vars)):
            if vars[i] in clause_vars and min(probabilities[i], 1 - probabilities[i]) < min(close_probability,
                                                                                            1 - close_probability):
                variable = vars[i]
                close_probability = probabilities[i]
        return variable

    def maximumPosterioriEstimationHeuristic() -> str:

        clause_vars = {literal.lower() for disjunction in clause for literal in disjunction}
        max_var = None
        max_score = 0
        for i in range(len(vars)):
            if vars[i] in clause_vars:
                prob_true = probabilities[i]
                prob_false = 1 - prob_true
                num_satisfied_true = len([disjunction
                                          for disjunction in clause
                                          if any(literal == vars[i]
                                                 for literal in disjunction)])

                num_satisfied_false = len([disjunction for disjunction in clause
                                           if any(literal == vars[i].upper()
                                                  for literal in disjunction)])

                score = prob_true ** num_satisfied_true * prob_false ** num_satisfied_false

                if score > max_score:
                    max_score = score
                    max_var = vars[i]

        return max_var

    def mostConstrainedVariableHeuristic():
        var_counts = {}
        for disjunction in clause:
            for literal in disjunction:
                var_counts[literal.lower()] = var_counts.get(literal.lower(), 0) + 1
        sorted_vars = sorted(var_counts.items(), key=lambda x: x[1])
        return sorted_vars[0][0]

    def JeroslowWangHeuristic() -> str:
        jw_scores = {}
        for disjunction in clause:
            for literal in disjunction:
                var = literal.lower()
                jw_scores[var] = jw_scores.get(var, 0) + probabilities[vars.index(var)] * 2 ** (-len(clause))
        return max(jw_scores, key=jw_scores.get)

    def standartDPLL() -> str:
        for disjunction in clause:
            for literal in disjunction:
                return literal.lower()
    s = ''
    match heuristic_type:
        case HeuristicType.STANDART_DPLL:
            s = standartDPLL()
        case HeuristicType.MAXIMUM_LIKELIHOOD_ESTIMATION:
            s = maximumLikelihoodEstimationHeuristic()
        case HeuristicType.MAXIMUM_POSTERIORI_ESTIMATION:
            s = maximumLikelihoodEstimationHeuristic()
        case HeuristicType.MOST_CONSTRAINED_VARIABLE:
            s = mostConstrainedVariableHeuristic()
        case HeuristicType.JEROSLOW_WANG:
            s = JeroslowWangHeuristic()
    return NextRec(s, RecType.BOTH)


def recalculate_probabilities(clauses: list[list[str]], probabilities: list[float], variables: list[str], variable: str,
                              is_false=False) -> list[float]:
    relevant_clauses = []
    for clause in clauses:
        if is_false:
            if variable.upper() in clause:
                relevant_clauses.append(clause)
        else:
            if variable in clause:
                relevant_clauses.append(clause)

    product_probabilities = 1
    for clause in relevant_clauses:
        for literal in clause:
            if is_false:
                if literal == variable.upper():
                    product_probabilities *= (1 - probabilities[variables.index(variable)])
                else:
                    product_probabilities *= probabilities[variables.index(literal.lower())]
            else:
                if literal == variable:
                    product_probabilities *= probabilities[variables.index(literal)]

    updated_probabilities = [0] * len(probabilities)
    for i in range(len(variables)):
        if variable == variables[i]:
            if is_false:
                updated_probabilities[i] = 1 - probabilities[i]
            else:
                updated_probabilities[i] = 1
        elif variables[i] in [literal[1:] for literal in relevant_clauses]:
            if is_false:
                if variables[i].upper() in [literal for clause in relevant_clauses for literal in clause]:
                    updated_probabilities[i] = probabilities[i] * (1 - product_probabilities)
                else:
                    updated_probabilities[i] = probabilities[i] * product_probabilities
            else:
                updated_probabilities[i] = probabilities[i] * product_probabilities
        else:
            updated_probabilities[i] = probabilities[i]

    total_prob = sum(updated_probabilities)
    for i in range(len(updated_probabilities)):
        updated_probabilities[i] /= total_prob

    return updated_probabilities


def gen_branch(clause, next_rec, bool):
    branch = []

    if bool:
        for lit in clause:
            if next_rec.var in lit:
                continue
            temp = lit.copy()
            if (biba := next_rec.var.upper()) in temp:
                temp.remove(biba)

            if temp not in branch:
                branch.append(temp)
    else:
        for lit in clause:
            if next_rec.var.upper() in lit:
                continue
            temp = lit.copy()
            if (biba := next_rec.var) in temp:
                temp.remove(biba)

            if temp not in branch:
                branch.append(temp)

    return branch


def set_inter(inter, vars, var, bool):
    for i in range(len(vars)):
        if vars[i] == var:
            inter[i] = 1 if bool else 0


def base_dpll(clause, vars, inter, probabilities):
    def trueBrunch() -> bool:
        true_branch = gen_branch(clause, next_rec, True)
        set_inter(inter, vars, next_rec.var, True)

        # if heuristic_type == HeuristicType.STANDART_DPLL:
        #     if base_dpll(true_branch, vars, inter, probabilities):
        #         return True
        # else:
        #     if base_dpll(true_branch, vars, inter, recalculate_probabilities(clause, probabilities, vars, next_rec.var)):
        #         return True
        if base_dpll(true_branch, vars, inter, recalculate_probabilities(clause, probabilities, vars, next_rec.var)):
            return True

    def falseBrunch() -> bool:
        false_branch = gen_branch(clause, next_rec, False)
        set_inter(inter, vars, next_rec.var, False)

        # if heuristic_type == HeuristicType.STANDART_DPLL:
        #     if base_dpll(false_branch, vars, inter, probabilities):
        #         return True
        # else:
        #     if base_dpll(false_branch, vars, inter, recalculate_probabilities(clause, probabilities, vars, next_rec.var, is_false=True)):
        #         return True
        if base_dpll(false_branch, vars, inter, recalculate_probabilities(clause, probabilities, vars, next_rec.var, is_false=True)):
            return True

    if len(clause) == 0:
        with open(output_file, 'a', encoding='UTF-8') as f:
            f.write('|{:<35}|'.format(heuristic_type.name))
            for value in inter:
                if value is not None:
                    f.write('{:<6}|'.format(value))
                else:
                    f.write('{:<6}|'.format('Any'))
            f.write('\n')
            # f.write(str(inter) + '\n')
            for i in range(len(vars)*7 + 37):
                f.write('-')
            f.write('\n')
        return True
    else:
        for lit in clause:
            if not lit:
                return False
    # clause = simplify_cnf(clause)
    next_rec = get_next_rec(clause, vars, probabilities)

    if probabilities[vars.index(next_rec.var)] >= 0.5:
        if (next_rec.type == RecType.TRUE or next_rec.type == RecType.BOTH) and trueBrunch():
            return True
        if next_rec.type == RecType.FALSE or next_rec.type == RecType.BOTH and falseBrunch():
            return True

    else:
        if next_rec.type == RecType.FALSE or next_rec.type == RecType.BOTH and falseBrunch():
            return True
        if (next_rec.type == RecType.TRUE or next_rec.type == RecType.BOTH) and trueBrunch():
            return True

    return False


def dpll(clause, vars):
    inter = []
    probabilities = []
    for _ in vars:
        probabilities.append(0.5)
        inter.append(None)
    base_dpll(clause, vars, inter, probabilities)
    return


def executeCommand():
    def set_heuristic_type(type: str):
        global heuristic_type
        match type:
            case HeuristicType.STANDART_DPLL.value:
                heuristic_type = HeuristicType.STANDART_DPLL
            case HeuristicType.MAXIMUM_LIKELIHOOD_ESTIMATION.value:
                heuristic_type = HeuristicType.MAXIMUM_LIKELIHOOD_ESTIMATION
            case HeuristicType.MAXIMUM_POSTERIORI_ESTIMATION.value:
                heuristic_type = HeuristicType.MAXIMUM_POSTERIORI_ESTIMATION
            case HeuristicType.MOST_CONSTRAINED_VARIABLE.value:
                heuristic_type = HeuristicType.MOST_CONSTRAINED_VARIABLE
            case HeuristicType.JEROSLOW_WANG.value:
                heuristic_type = HeuristicType.JEROSLOW_WANG
            case 'ALL':
                heuristic_type = 'ALL'
            case _:
                sys.exit(type + ''' - Invalid heuristic type.
There are only 6 types of heuristics available to work with:
ST: STANDART_DPLL
MLE: MAXIMUM_LIKELIHOOD_ESTIMATION
MPE: MAXIMUM_POSTERIORI_ESTIMATION
MCV: MOST_CONSTRAINED_VARIABLE
JW: JEROSLOW_WANG
ALL: all kinds of heuristics in order''')

    def set_input_file(file: str):
        try:
            f = open(file)
            f.close()
        except IOError:
            sys.exit('The file does not exist or there is no access to it')
        global input_file
        input_file = file
    def set_output_file(file: str):
        try:
            f = open(file)
            f.close()
        except IOError:
            sys.exit('Invalid file name')
        global output_file
        output_file = file

    for i in range(1, len(sys.argv), 2):
        if len(sys.argv) - 1 == i:
            sys.exit('There is no argument for the command ' + sys.argv[i])
        match sys.argv[i]:
            case '-heuristic':
                set_heuristic_type(sys.argv[i + 1])
            case '-input':
                set_input_file(sys.argv[i + 1])
            case '-output':
                set_output_file(sys.argv[i + 1])
            case _:
                sys.exit('Invalid command ' + sys.argv[i])
def main():
    global heuristic_type
    global input_file
    global output_file
    if (len(sys.argv) == 2) and (sys.argv[1] == '--help'):
        sys.exit(open('Help').read())
    else:
        if len(sys.argv) > 1:
            executeCommand()

        # print(sys.argv)
    with open(input_file, 'r', encoding='UTF-8') as f:
        args = f.readline().strip().split()
        g_args = []
        g_vars = []
        for arg in args:
            nums_of_arg = arg.lower().strip().split('x')[1:]
            xes = [c for c in arg if c == 'X' or c == 'x']
            g_args.append([a[0] + a[1] for a in zip(xes, nums_of_arg)])
            g_vars.extend([a[0] + a[1] for a in zip([c.lower() for c in arg if c == 'X' or c == 'x'], nums_of_arg)])
        g_vars = sorted(list(set(g_vars)))

    with open(output_file, 'w', encoding='UTF-8') as f:
        for i in range(len(g_vars) * 7 + 37):
            f.write('-')
        f.write('\n')
        f.write('|{:<35}|'.format('HEURISTIC'))
        for var in g_vars:
            f.write('{:<6}|'.format(var))
        f.write('\n')
        # f.write(str(vars) + '\n')
        for i in range(len(g_vars) * 7 + 37):
            f.write('-')
        f.write('\n')

    if heuristic_type == 'ALL':
        times = []
        for heuristic in HeuristicType:
            heuristic_type = heuristic
            t = perf_counter()

            dpll(g_args, g_vars)
            # times.append(perf_counter() - t)
            print("{:<35}".format(heuristic.name + ':'), perf_counter() - t, sep='')
    else:
        t = perf_counter()
        dpll(g_args, g_vars)
        print("{:<35}".format(heuristic_type.name + ':'), perf_counter() - t, sep='')


if __name__ == '__main__':
    main()