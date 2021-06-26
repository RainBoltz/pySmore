import numpy as np
from libs.util import fast_sigmoid

def get_dotproduct_loss(from_embedding, to_embedding, weight):
    weight = float(weight)
    prediction = np.dot(from_embedding, to_embedding.T)
    gradient = weight - prediction

    from_loss = gradient * to_embedding
    to_loss = gradient * from_embedding

    return from_loss, to_loss

def get_loglikelihood_loss(from_embedding, to_embedding, weight):
    weight = float(weight)
    prediction = np.dot(from_embedding, to_embedding.T)
    gradient = weight - fast_sigmoid(prediction)

    from_loss = gradient * to_embedding
    to_loss = gradient * from_embedding
    
    return from_loss, to_loss

def get_margin_bpr_loss(from_embedding, to_embedding_pos, to_embedding_neg, margin=8.0):
    diff_to_embedding = to_embedding_pos - to_embedding_neg
    prediction = np.dot(from_embedding, diff_to_embedding.T) - margin
    gradient = 0.0 - fast_sigmoid(prediction)

    from_loss = gradient * diff_to_embedding
    to_loss_pos = gradient * from_embedding
    to_loss_neg = -gradient * from_embedding
    
    return from_loss, to_loss_pos, to_loss_neg


def get_bpr_loss(from_embedding, to_embedding_pos, to_embedding_neg):
    diff_to_embedding = to_embedding_pos - to_embedding_neg
    prediction = np.dot(from_embedding, diff_to_embedding.T)
    gradient = 0.0 - fast_sigmoid(prediction)

    from_loss = gradient * diff_to_embedding
    to_loss_pos = gradient * from_embedding
    to_loss_neg = to_loss_pos
    
    return from_loss, to_loss_pos, to_loss_neg

#TODO
def get_convolutional_loss(from_embedding, to_embedding_pos, to_embedding_negs, margin=8.0):
    diff_to_embedding = to_embedding_pos - to_embedding_negs
    prediction = np.dot(from_embedding, diff_to_embedding.T) - margin
    gradient = np.apply_along_axis(lambda x: 0.0 - fast_sigmoid(x), axis=0, arr=prediction)

    from_loss = gradient * diff_to_embedding
    to_loss_pos = gradient * from_embedding
    to_loss_negs = -gradient * from_embedding
    
    return from_loss, to_loss_pos, to_loss_negs