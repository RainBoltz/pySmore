import numpy as np
import multiprocessing as mp



class embeddings:
    def __init__(self, amount, dimensions=64):
        self.manager = mp.Manager()

        random_generator = np.random.default_rng()
        self.embed = self.manager.list([
            self.manager.list(random_generator.uniform(-0.5, 0.5, dimensions))
            for _ in range(amount)
        ])

    def get_embedding(self, idx, lock=False):
        if lock:
            lock = self.manager.Lock()

        if type(idx) == int:
            return self.embed[idx], lock
        elif type(idx) == str:

            return 