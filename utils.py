import numpy as np
import random

'''
include some helper functions
'''
MAX_ITERATION = 10

def calc_fitness_score(n_a, k):
    if n_a <= k:
        return 1 - n_a/k
    else:
        return (1+n_a)/k - 1

# copied from https://github.com/hardmaru/estool/blob/master/simple_es_example.ipynb
def test_solver(solver, fit_func):
    history = []
    for j in range(MAX_ITERATION):
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = fit_func(solutions[i])
        solver.tell(fitness_list)
        result = solver.result() # first element is the best solution, second element is the best fitness
        history.append(result[1])
        if (j+1) % 5 == 0:
            print("fitness at iteration", (j+1), result[1])
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

def to_zero_one(logits):
    out_logits = np.zeros(logits.shape)
    out_logits[logits>0.5] = 1
    return out_logits

def random_action(l):
    out = []
    for _ in range(l):
        rand_num = random.randint(0,1)
        out.append(rand_num)
    return out

def main():
    out = to_zero_one(np.array([0.3, 0.6, 0.1, 0.5]))
    print(out)
    out = random_action(5)
    print(out)

main()