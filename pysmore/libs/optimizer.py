import numpy as np
from .util import fast_sigmoid

def get_dotproduct_loss(from_embedding, to_embedding, weight):
    prediction = np.dot(from_embedding, to_embedding.T)
    gradient = (weight - prediction).sum()

    from_loss = gradient * to_embedding
    to_loss = gradient * from_embedding

    return from_loss, to_loss

def get_loglikelihood_loss(from_embedding, to_embedding, weight):
    prediction = np.dot(from_embedding, to_embedding.T)
    gradient = (weight - fast_sigmoid(prediction)).sum()

    from_loss = gradient * to_embedding
    to_loss = gradient * from_embedding
    
    return from_loss, to_loss

def get_margin_bpr_loss(from_embedding, to_embedding_pos, to_embedding_negs, margin=8.0):
    diff_to_embedding = to_embedding_pos - to_embedding_negs
    prediction = np.dot(from_embedding, diff_to_embedding.T) - margin
    gradient = (0.0 - fast_sigmoid(prediction)).sum()

    from_loss = gradient * diff_to_embedding
    to_loss_pos = gradient * from_embedding
    to_loss_negs = -gradient * from_embedding
    
    return from_loss, to_loss_pos, to_loss_negs


def get_bpr_loss(from_embedding, to_embedding_pos, to_embedding_negs, margin=8.0):
    diff_to_embedding = to_embedding_pos - to_embedding_negs
    prediction = np.dot(from_embedding, diff_to_embedding.T) - margin
    gradient = (0.0 - fast_sigmoid(prediction)).sum()

    from_loss = gradient * diff_to_embedding
    to_loss_pos = gradient * from_embedding
    to_loss_negs = to_loss_pos
    
    return from_loss, to_loss_pos, to_loss_negs
