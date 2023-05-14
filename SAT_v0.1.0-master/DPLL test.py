import time
from DPLL import dpll
import os
import sys

def run_dpll(input_file):
    start_time = time.time()
    with open(input_file) as f:
        g_args = f.readline().strip().split()

    g_vars = sorted(list(set([c.lower() for arg in g_args for c in arg])))

    dpll(g_args, g_vars)

    elapsed_time = time.time() - start_time
    return elapsed_time

if __name__ == '__main__':
    num_tests = int(input())
    for i in range(1, num_tests+1):
        input_file = f"input{i}.txt"
        if not os.path.isfile(input_file):
            print(f"Error: input file {input_file} does not exist.")
            sys.exit()
        print(f"Running test {i} with input file {input_file}")
        elapsed_time = run_dpll(input_file)
        print(f"Elapsed time: {elapsed_time:.4f} seconds")
