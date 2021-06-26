import numpy as np
import multiprocessing as mp
import ctypes

def create_embeddings(amount, dimensions=64, lock=True):
    shared_array_base = mp.Array(ctypes.c_double, amount * dimensions, lock=lock)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(amount, dimensions)
    shared_array[:,:] = np.random.uniform(low=-0.5, high=0.5, size=(amount, dimensions))/dimensions
    return shared_array

def create_embeddings_unsafe(amount, dimensions=64):
    shared_array_base = mp.RawArray(ctypes.c_double, amount * dimensions)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(amount, dimensions)
    shared_array[:,:] = np.random.uniform(low=-0.5, high=0.5, size=(amount, dimensions))/dimensions
    return shared_array

def update_embedding(source_matrix, idx, target_matrix, learning_rate):
    source_matrix[idx] += learning_rate * target_matrix

def update_l2_embedding(source_matrix, idx, target_matrix, learning_rate, regularize_ratio=1e-4):
    source_matrix[idx] += (learning_rate * target_matrix - regularize_ratio * source_matrix[idx])

def save_embeddings(embeddings, indices, fname):
    print('saving files (%s)...'%fname, end='', flush=True)
    np.savetxt(fname+".rep", embeddings, fmt='%.8f', delimiter=' ')
    with open(fname+".idx", 'w') as f:
        f.write("%s\n"%('\n'.join(indices)))
    print('DONE', flush=True)