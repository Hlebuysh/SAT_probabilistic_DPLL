input_file = open('input.txt')
output_file = open('output.txt')
args = input_file.readline().strip().split()
g_args = []
g_vars = []
values = []
for arg in args:
    nums_of_arg = arg.lower().strip().split('x')[1:]
    xes = [c for c in arg if c == 'X' or c == 'x']
    g_args.append([a[0] + a[1] for a in zip(xes, nums_of_arg)])
g_vars = [i[1:-2] for i in output_file.readline()[1:-1].strip().split()]
values = [int(i[:-1]) if i[0] != 'N' else None for i in output_file.readline()[1:-1].strip().split()]


result_expression = []
ix = 0
for disjunction in g_args:
    result_expression.append([])
    for literal in disjunction:
        if literal[0] == 'x':
            if (values[g_vars.index(literal)] is None) or values[g_vars.index(literal)] == 0:
                result_expression[ix].append(0)
            else:
                result_expression[ix].append(1)
        else:
            if (values[g_vars.index(literal.lower())] is None) or values[g_vars.index(literal.lower())] == 0:
                result_expression[ix].append(1)
            else:
                result_expression[ix].append(0)
    ix += 1
for disjunction in result_expression:
    if sum(disjunction) == 0:
        print('ERROR')
        break
print('END')
