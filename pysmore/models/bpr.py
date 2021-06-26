from pysmore.libs import graph, optimizer, embedding, util
import multiprocessing as mp

### global variables ###
globalVariables = {
    'graph':        None,
    'optimizer':    optimizer.get_margin_bpr_loss,
    'updater':      embedding.update_l2_embedding,
    'progress':     util.print_progress,
    'l2_reg':       0.0001,
    'init_lr':      0.025,
    'min_lr':       0.025 * 1e-4,
    'n_neg':        5
}

current_update_times =  mp.RawValue('i', 0)
userEmbed =             None
itemEmbed =             None
######


### main learner ###
def learner():
    globalVariables['graph'].cache_edge_samples(globalVariables['worker_update_times'])
    globalVariables['progress'](0.0)
    _learning_rate = globalVariables['init_lr']
    for i in range(1, globalVariables['worker_update_times']+1):
        user, user_idx, item_pos, item_pos_idx, weight = \
            globalVariables['graph'].draw_an_edge_from_sample()

        for j in range(globalVariables['n_neg']):
            item_neg, item_neg_idx = globalVariables['graph'].draw_an_item_uniformly()

            user_embedding = userEmbed[user_idx]
            item_pos_embedding = itemEmbed[item_pos_idx]
            item_neg_embedding = itemEmbed[item_neg_idx]

            user_loss, item_pos_loss, item_neg_loss = \
                globalVariables['optimizer'](user_embedding, item_pos_embedding, item_neg_embedding)
            
            current_progress_percentage = current_update_times.value / globalVariables['total_update_times']
            globalVariables['updater'](userEmbed, user_idx, user_loss, _learning_rate, globalVariables['l2_reg'])
            globalVariables['updater'](itemEmbed, item_pos_idx, item_pos_loss, _learning_rate, globalVariables['l2_reg'])
            globalVariables['updater'](itemEmbed, item_neg_idx, item_neg_loss, _learning_rate, globalVariables['l2_reg'])

        monitor_flag = int(1e3)
        if i % monitor_flag == 0:
            _learning_rate = globalVariables['init_lr'] * (1.0 - current_progress_percentage)
            _learning_rate = max(globalVariables['min_lr'], _learning_rate)
            current_update_times.value += monitor_flag
            globalVariables['progress'](current_progress_percentage)
######


### user functions ###
def create_graph(train_path, embedding_dimension=64):
    global globalVariables
    global userEmbed
    global itemEmbed

    globalVariables['graph'] = graph.Graph(train_path, mode='edge')

    print('create embeddings...', end='', flush=True)
    userEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].user_count,
        dimensions=embedding_dimension)
    itemEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].item_count,
        dimensions=embedding_dimension)
    print('DONE', flush=True)

def set_param(params):
    global globalVariables
    for key in params:
        globalVariables[key] = params[key]

def train(update_times=10, workers=1):
    global globalVariables
    globalVariables['total_update_times'] = int(update_times * 1000000)
    globalVariables['workers'] = workers
    globalVariables['worker_update_times'] = int((update_times * 1000000)/workers)
    globalVariables['min_lr'] = globalVariables['init_lr'] * 1000 / globalVariables['total_update_times']

    util.optimize_numpy_multiprocessing(workers)

    processes = []
    for i in range(workers):
        p = mp.Process(target=learner, args=())
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    current_update_times.value = 0
    globalVariables['progress'](1.0)
    
def save_embeddings(file_prefix="bpr"):
    global globalVariables
    global userEmbed
    global itemEmbed
    print()
    embedding.save_embeddings(userEmbed, globalVariables['graph'].users, file_prefix+'_users')
    embedding.save_embeddings(itemEmbed, globalVariables['graph'].items, file_prefix+'_items')
######
