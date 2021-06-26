import os
import math

SIGMOID_TABLE_SIZE = 1000
MAX_SIGMOID = 8.0
SIGMOID_TABLE = [ 
    1.0 / (1.0 + math.exp(i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID))
        for i in range(SIGMOID_TABLE_SIZE+1)
]

def graph_reader_generator(path, delimiter):
    with open(path) as f:
        for line in f:
            user, item, weight = line.rstrip().split(delimiter)
            yield user, item, weight

def optimize_numpy_multiprocessing(worker_amount):
    if worker_amount >= 4:
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

def print_progress(percentage):
    print('progress: %.2f%c'%(percentage*100,'%'), end='\r', flush=True)
