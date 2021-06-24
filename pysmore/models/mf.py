
from libs import graph, optimizer, embedding
import multiprocessing as mp
from tqdm import trange

# settings
train_path = r'D:\repository\pySmore\pysmore\data\kktv.ui.train.txt'
update_times = 10
init_learning_rate = 0.025
minimal_learning_rate = init_learning_rate * 1e-4
l2_regularize_ratio = 1e-2
workers = 4

# global constants
myGraph = graph.Graph(train_path, delimiter=' ')
myOptimizer = optimizer.get_dotproduct_loss
print('create embeddings...')
userEmb = embedding.Embeddings(myGraph.get_user_amount())
itemEmb = embedding.Embeddings(myGraph.get_item_amount())
total_update_times = int(update_times * 1e6)

# main learner
def learner(flag):
    global myGraph, myOptimizer
    global userEmb, itemEmb
    global total_update_times, init_learning_rate
    
    user, user_idx = myGraph.draw_user()
    item, item_idx = myGraph.draw_item(user)
    weight = myGraph.get_weight(user, item)

    user_embedding = userEmb.get_embedding(user_idx)
    item_embedding = itemEmb.get_embedding(item_idx)

    user_loss, item_loss = myOptimizer(user_embedding, item_embedding, weight)
    learning_rate = (1.0 - float(flag) / total_update_times)

    return user_idx, user_loss, item_idx, item_loss, learning_rate


def update(results):
    global userEmb, itemEmb
    global l2_regularize_ratio
    user_idx, user_loss, item_idx, item_loss, learning_rate = results
    userEmb.update_l2_embedding(user_idx, user_loss, learning_rate, l2_regularize_ratio)
    itemEmb.update_l2_embedding(item_idx, item_loss, learning_rate, l2_regularize_ratio)

def main():
    with mp.Pool(workers) as p:
        for i in trange(total_update_times):
            p.apply_async(learner, args = (i, ), callback=update)

main()

