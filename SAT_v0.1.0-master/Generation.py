from sympy import *
from sympy.logic.boolalg import to_cnf
import random


# Генерация случайного логического выражения
def generate_expression(variables):
    if len(variables) == 1:
        return variables[0]
    op = random.choice(['&', '|', '>>', '<<', '^'])
    if op == '&':
        return And(generate_expression(variables[:len(variables) // 2]), generate_expression(variables[len(variables) // 2:]))
    elif op == '|':
        return Or(generate_expression(variables[:len(variables) // 2]), generate_expression(variables[len(variables) // 2:]))
    elif op == '>>':
        return Implies(generate_expression(variables[:len(variables) // 2]), generate_expression(variables[len(variables) // 2:]))
    elif op == '<<':
        return Implies(generate_expression(variables[len(variables) // 2:]), generate_expression(variables[:len(variables) // 2]))
    elif op == '^':
        return Xor(generate_expression(variables[:len(variables) // 2]), generate_expression(variables[len(variables) // 2:]))


# Генерация случайной логической формулы в КНФ
def generate_knf_formula(num_vars, num_clauses, clause_size):
    variables = symbols('x:%d' % num_vars)
    clauses = []
    for _ in range(num_clauses):
        clause = []
        while len(clause) < clause_size:
            var = random.choice(variables)
            if random.random() < 0.5:
                var = Not(var)
            if var not in clause:
                clause.append(var)
        clauses.append(Or(*clause))
    expr = And(*clauses)
    cnf_expr = to_cnf(expr)
    return str(cnf_expr)


# Преобразование строки к нужному виду
def transform_formula(formula_str):
    # Удаляем пробелы
    formula_str = formula_str.replace(' ', '')
    # Удаляем символы "~" и "|"
    formula_str = formula_str.replace('~x', 'X').replace('|', '')
    # Заменяем символ "&" на пробел
    formula_str = formula_str.replace('&', ' ')
    formula_str = formula_str.replace('(', '').replace(')', '')
    return formula_str


# Пример использования
cnf_expr = generate_knf_formula(500, 700, 20)
print(cnf_expr)
print(transform_formula(str(cnf_expr)))
with open('input.txt', 'w+', encoding='UTF-8') as f:
    f.write(transform_formula(str(cnf_expr)))


def generate_cnf(vars_count: int, disjunctions_count: int, min_disjunction_variables=1, max_disjunction_variables=100):
    vars = ['x'+str(i) for i in range(vars_count)]
    result = ''
    for i in range(disjunctions_count):
        for j in range(random.randint(min_disjunction_variables, max_disjunction_variables)):
            result += random.choice(vars)
        result += ' '
    return result