import multiprocessing as mp
from ctypes import Structure, c_int, c_double

import numpy as np
import time


def updatez(e1,e2,loss):#,lock=None):
    #if lock:
    #    lock.acquire()
    n = len(loss)
    for i in range(n):
        e1[i] = e1[i] + loss[i]
        e2[i] = e2[i] - loss[i]
    #if lock:
    #    lock.release()
    


if __name__ == '__main__':

    manager = mp.Manager()

    random_generator = np.random.default_rng()
    ee1 = manager.list(random_generator.uniform(-0.5, 0.5, 5))
    ee2 = manager.list(random_generator.uniform(-0.5, 0.5, 5))
    ee3 = manager.list(random_generator.uniform(-0.5, 0.5, 5))
    

    #print('q:', list(ee1))
    #print('p:', list(ee3))
    #print()
    #print('p*100 + q:', list(np.array(list(ee1)) + np.array(list(ee3)) * 100))

    s = time.perf_counter()
    with mp.Pool() as pool:
        
        for i in range(1000):
            pool.apply_async(updatez, args=(ee1,ee2,ee3))
            
        pool.close()
        pool.join()
    e = time.perf_counter()
    print("without lock: %f"%(e-s))

'''
    s = time.perf_counter()
    l = manager.Lock()
    with mp.Pool() as pool:
        
        for i in range(1000):
            pool.apply_async(updatez, args=(ee1,ee2,ee3,l))
            
        pool.close()
        pool.join()
    e = time.perf_counter()
    print("with lock: %f"%(e-s))
'''
    

    #print('a:', ee1[:])