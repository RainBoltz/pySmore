import multiprocessing as mp
import numpy as np
import ctypes


def create_embeddings_unsafe(amount, dimensions=3):
    shared_array_base = mp.RawArray(ctypes.c_double, amount * dimensions)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(amount, dimensions)
    shared_array[:,:] = np.full((amount, dimensions), 1)
    return shared_array

A = create_embeddings_unsafe(amount=3)
B = create_embeddings_unsafe(amount=3)

def f(x):
    A[int(x/3), x%3] = x
    print(A)
    print(A[[0,1]])

with mp.Pool(4) as p:
    rs = []
    for i in range(3*3):
        p.apply_async(f, args=(i,))
    print('all workers running...')
    p.close()
    print('pool closed...')
    p.join()
    print('pool joined...')
    
    
    