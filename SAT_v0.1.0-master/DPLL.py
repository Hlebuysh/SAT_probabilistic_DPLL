import sys
from time import perf_counter
from DPLL_classes import *


heuristic_type = 'ALL'
input_file = 'input.txt'
output_file = 'output.txt'
command_line_expression = False
iterations = 0
variables = []
values = []


def compare_variables(var1: str, var2: str) -> bool:
    return int(var1[1:]) < int(var2[1:])


def sort_variables():
    global variables
    variables = sorted(variables, key=lambda x: int(x[1:]))


def get_variable_index(name: str) -> int | None:
    return int(name[1:])
    start_element = 0
    end_element = len(variables) - 1
    n = int(name[1:])
    while start_element <= end_element:
        middle_element = start_element + (end_element - start_element) // 2
        if variables[middle_element] == name:
            return middle_element
        elif compare_variables(variables[middle_element], name):
            start_element = middle_element + 1
        else:
            end_element = middle_element - 1
    return None


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


def get_next_rec(clause: list[list[str]], probabilities: list[float]) -> NextRec:
    def maximumLikelihoodEstimationHeuristic() -> str:
        max_variable = None
        max_probability = -1

        for disjunction in clause:
            for literal in disjunction:
                var = literal.lower()
                probability = probabilities[get_variable_index(var)]
                if probability > max_probability:
                    max_probability = probability
                    max_variable = var

        return max_variable
    def maximumPosterioriEstimationHeuristic() -> str:
        clause_vars = {literal.lower() for disjunction in clause for literal in disjunction}

        variable = variables[0]
        close_probability = probabilities[0]
        for i in range(1, len(variables)):
            if variables[i] in clause_vars and min(probabilities[i], 1 - probabilities[i]) < min(close_probability,
                                                                                            1 - close_probability):
                variable = variables[i]
                close_probability = probabilities[i]
        return variable

    def mostConstrainedVariableHeuristic():
        var_counts = [0] * len(variables)
        for disjunction in clause:
            for literal in disjunction:
                ix = get_variable_index(literal.lower())
                var_counts[ix] += 1
        i_min = 0
        while var_counts[i_min] == 0:
            i_min += 1
        for i in range(i_min, len(var_counts)):
            if (var_counts[i] != 0) and (var_counts[i] < var_counts[i_min]):
                i_min = i
        return variables[i_min]

    def JeroslowWangHeuristic() -> str:
        jw_scores = [0] * len(variables)
        for disjunction in clause:
            for literal in disjunction:
                ix = get_variable_index(literal.lower())
                jw_scores[ix] += probabilities[ix] / len(clause)
                # jw_scores[ix] += probabilities[ix] * 2 ** (10 - len(clause))
        return variables[max(enumerate(jw_scores), key=lambda x: x[1])[0]]

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
            s = maximumPosterioriEstimationHeuristic()
        case HeuristicType.MOST_CONSTRAINED_VARIABLE:
            s = mostConstrainedVariableHeuristic()
        case HeuristicType.JEROSLOW_WANG:
            s = JeroslowWangHeuristic()
    return NextRec(s, RecType.BOTH)


def recalculate_probabilities(clauses: list[list[str]], probabilities: list[float], variable: str,
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


def set_inter(var: str, is_true_brunch: bool):
    values[get_variable_index(var)] = 1 if is_true_brunch else 0
    

def reset_inter(var):
    values[get_variable_index(var)] = None


def base_dpll(clause, probabilities):
    global variables
    global values
    global iterations
    iterations += 1
    def trueBrunch() -> bool:
        true_branch = gen_branch(clause, next_rec, True)
        set_inter(next_rec.var, True)

        # if heuristic_type == HeuristicType.STANDART_DPLL:
        #     if base_dpll(true_branch, probabilities):
        #         return True
        # else:
        #     if base_dpll(true_branch, recalculate_probabilities(clause, probabilities, next_rec.var)):
        #         return True
        if base_dpll(true_branch, recalculate_probabilities(clause, probabilities, next_rec.var)):
            return True
        reset_inter(next_rec.var)

    def falseBrunch() -> bool:
        false_branch = gen_branch(clause, next_rec, False)
        set_inter(next_rec.var, False)

        # if heuristic_type == HeuristicType.STANDART_DPLL:
        #     if base_dpll(false_branch, probabilities):
        #         return True
        # else:
        #     if base_dpll(false_branch, recalculate_probabilities(clause, probabilities, next_rec.var, is_false=True)):
        #         return True
        if base_dpll(false_branch, recalculate_probabilities(clause, probabilities, next_rec.var, is_false=True)):
            return True
        reset_inter(next_rec.var)

    if len(clause) == 0:
        with open(output_file, 'a', encoding='UTF-8') as f:
            f.write('|{:<35}|'.format(heuristic_type.name if type(heuristic_type) == HeuristicType else 'ALL'))
            for value in values:
                if value is not None:
                    f.write('{:<6}|'.format(value))
                else:
                    f.write('{:<6}|'.format('Any'))
            f.write('\n')
            # f.write(str(values) + '\n')
            for i in range(len(variables)*7 + 37):
                f.write('-')
            f.write('\n')
        return True
    else:
        for lit in clause:
            if not lit:
                return False
    # clause = simplify_cnf(clause)
    next_rec = get_next_rec(clause, probabilities)

    if probabilities[get_variable_index(next_rec.var)] > 0.5:
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


def dpll(clause):
    global values
    values = [None] * len(variables)
    probabilities = [0.5] * len(variables)
    base_dpll(clause, probabilities)
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

    def set_command_line_expression():
        global command_line_expression
        command_line_expression = True

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-expression':
            set_command_line_expression()
            i += 1
            continue

        if len(sys.argv) - 1 == i:
            sys.exit('There is no argument for the command ' + sys.argv[i])
        match sys.argv[i]:
            case '-heuristic':
                set_heuristic_type(sys.argv[i + 1])
                i += 2
            case '-input':
                set_input_file(sys.argv[i + 1])
                i += 2
            case '-output':
                set_output_file(sys.argv[i + 1])
                i += 2
            case _:
                sys.exit('Invalid command ' + sys.argv[i])


def main():
    global heuristic_type
    global input_file
    global output_file
    global command_line_expression
    global iterations
    global variables
    
    if (len(sys.argv) == 2) and (sys.argv[1] == '--help'):
        sys.exit(open('Help').read())
    else:
        if len(sys.argv) > 1:
            executeCommand()

    if command_line_expression is True:
        expression = input('Expression:\n')
        with open(input_file, 'w', encoding='UTF-8') as f:
            f.write(expression)
        # print(sys.argv)
    with open(input_file, 'r', encoding='UTF-8') as f:
        args = list(set(f.readline().strip().split()))
        g_args = []
        g_vars = []
        for arg in args:
            nums_of_arg = arg.lower().strip().split('x')[1:]
            xes = [c for c in arg if c == 'X' or c == 'x']
            g_args.append([a[0] + a[1] for a in zip(xes, nums_of_arg)])
            g_vars.extend([a[0] + a[1] for a in zip([c.lower() for c in arg if c == 'X' or c == 'x'], nums_of_arg)])
        g_vars = list(set(g_vars))
        variables = g_vars
        sort_variables()
        g_vars = variables
        variables = ['x'+str(i) for i in range(len(g_vars))]

    with open(output_file, 'w', encoding='UTF-8') as f:
        for i in range(len(variables) * 7 + 37):
            f.write('-')
        f.write('\n')
        f.write('|{:<35}|'.format('HEURISTIC'))
        for var in g_vars:
            f.write('{:<6}|'.format(var))
        f.write('\n')
        # f.write(str(variables) + '\n')
        for i in range(len(variables) * 7 + 37):
            f.write('-')
        f.write('\n')

    for i in range(len(g_args)):
        for j in range(len(g_args[i])):
            if g_args[i][j][0] == 'x':
                g_args[i][j] = variables[g_vars.index(g_args[i][j])]
            else:
                g_args[i][j] = variables[g_vars.index(g_args[i][j].lower())].upper()

    if heuristic_type == 'ALL':
        for heuristic in HeuristicType:
            iterations = 0
            heuristic_type = heuristic
            t = perf_counter()

            dpll(g_args)
            # times.append(perf_counter() - t)
            print("{:<35}{:<15.5f}".format(heuristic.name + ':', perf_counter() - t), iterations, sep='')
    else:
        iterations = 0
        t = perf_counter()
        dpll(g_args)
        print("{:<35}{:<15.5f}".format(heuristic_type.name + ':', perf_counter() - t), iterations, sep='')


if __name__ == '__main__':
    main()