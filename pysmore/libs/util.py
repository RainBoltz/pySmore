import os
import math
from numba import jit


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

@jit(nopython=True, fastmath=True)
def fast_sigmoid(x):
    MAX_SIGMOID = 8.0

    if x < -MAX_SIGMOID:
        return 0.0
    elif x > MAX_SIGMOID:
        return 1.0
    else:
        return 1.0 / (1.0 + math.exp(-x))


def print_progress(percentage):
    print('progress: %.4f%c'%(percentage*100,'%'), end='\r', flush=True)

def turn_on_debug_mode():
    import multiprocessing
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)