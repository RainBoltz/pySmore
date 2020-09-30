import numpy as np
import multiprocessing as mp
from ctypes import Structure, c_int, c_double



class embeddings:
    def __init__(self, amount, dimensions=64):
        self.manager = mp.Manager()
        random_generator = np.random.default_rng()
        self.embed = self.manager.list([
            mp.Array(c_double, random_generator.uniform(-0.5, 0.5, dimensions), lock=True)
            for _ in range(amount)
        ])
    
    