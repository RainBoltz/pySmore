import multiprocessing as mp
from ctypes import Structure, c_int, c_double

import numpy as np


def updatez(e1,e2,loss):
    print("GOGO!")
    n = len(loss)
    for i in range(n):
        e1[i] = e1[i] + loss[i]
        e2[i] = e2[i] - loss[i]
    print("DONE")
    
def init()

if __name__ == '__main__':
    mp.freeze_support()

    random_generator = np.random.default_rng()
    ee1 = mp.Array(c_double, random_generator.uniform(-0.5, 0.5, 5), lock=True)
    ee2 = mp.Array(c_double, random_generator.uniform(-0.5, 0.5, 5), lock=True)
    ee3 = mp.Array(c_double, random_generator.uniform(-0.5, 0.5, 5), lock=True)
    

    print(list(ee1))
    print(list(ee2))
    print(list(ee3))

    updatez(ee1,ee2,ee3)

    print(list(ee1))
    print(list(ee2))
    print(list(ee3))


    with mp.Pool(4, initializer=init, initargs=(a, v)) as pool:
        for i in range(100):
            pool.apply_async(updatez, args=(ee1,ee2,ee3))
            if i >= 90: print(i)
            
        pool.close()
        pool.join()

    print(list(ee1))
    print(list(ee2))
    print(list(ee3))