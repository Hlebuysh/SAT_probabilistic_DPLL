from enum import Enum, auto
from time import perf_counter


class RecType(Enum):
    FALSE = auto()
    TRUE = auto()
    BOTH = auto()


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


    return NextRec(JeroslowWangHeuristic(), RecType.BOTH)


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

            if not (temp in branch):
                branch.append(temp)
    else:
        for lit in clause:
            if next_rec.var.upper() in lit:
                continue
            temp = lit.copy()
            if (biba := next_rec.var) in temp:
                temp.remove(biba)

            if not (temp in branch):
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

        if base_dpll(true_branch, vars, inter, recalculate_probabilities(clause, probabilities, vars, next_rec.var)):
            return True
    def falseBrunch() -> bool:
        false_branch = gen_branch(clause, next_rec, False)
        set_inter(inter, vars, next_rec.var, False)

        if base_dpll(false_branch, vars, inter,
                     recalculate_probabilities(clause, probabilities, vars, next_rec.var, is_false=True)):
            return True

    if len(clause) == 0:
        with open("output.txt", 'w', encoding='UTF-8') as f:
            f.write(str(vars) + '\n')
            f.write(str(inter) + '\n')
        return True
    else:
        for lit in clause:
            if not lit:
                return False

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


def main():
    with open('input.txt', encoding='UTF-8') as f:
        args = f.readline().strip().split()
        g_args = []
        g_vars = []
        for arg in args:
            nums_of_arg = arg.lower().strip().split('x')[1:]
            xes = [c for c in arg if c == 'X' or c == 'x']
            g_args.append([a[0] + a[1] for a in zip(xes, nums_of_arg)])
            g_vars.extend([a[0] + a[1] for a in zip([c.lower() for c in arg if c == 'X' or c == 'x'], nums_of_arg)])
        g_vars = sorted(list(set(g_vars)))
    t = perf_counter()
    dpll(g_args, g_vars)
    print(perf_counter() - t)


if __name__ == '__main__':
    main()
