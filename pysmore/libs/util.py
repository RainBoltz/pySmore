import os
import numpy as np

RNG_GENERATOR = np.random.default_rng()
SIGMOID_TABLE_SIZE = 1000
MAX_SIGMOID = 8.0
SIGMOID_TABLE = [ 1.0 / (1.0 + np.exp(i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID)) for i in range(SIGMOID_TABLE_SIZE+1) ]


def graph_reader_generator(path, delimiter):
    with open(path) as f:
        for line in f:
            user, item, weight = line.rstrip().split(delimiter)
            yield user, item, weight


def fast_choice(pool_range, amount=1):
    global RNG_GENERATOR
    #amount = min(pool_range, amount)
    output = RNG_GENERATOR.choice(pool_range, amount, replace=False)
    return output

def optimize_numpy_multiprocessing():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def fast_sigmoid(x):
    global MAX_SIGMOID
    global SIGMOID_TABLE_SIZE
    global SIGMOID_TABLE

    if x < -MAX_SIGMOID:
        return 0.0
    elif x > MAX_SIGMOID:
        return 1.0
    else:
        return SIGMOID_TABLE[ int((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2) ]