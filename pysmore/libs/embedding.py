import numpy as np

class Embeddings:
    def __init__(self, amount, dimensions=64):
        self.embedding = np.random.uniform(low=-0.5, high=0.5, size=(amount, dimensions))

    def get_embedding(self, idx):
        return self.embedding[idx]

    def update_embedding(self, idx, matrix, learning_rate):
        self.embedding[idx] += learning_rate*matrix

    def update_l2_embedding(self, idx, matrix, learning_rate, regularize_ratio=1e-4):
        self.embedding[idx] += (learning_rate*matrix - regularize_ratio*self.embedding[idx])
    