from pysmore.libs import graph, optimizer, embedding, util
import multiprocessing as mp

### global variables ###
globalVariables = {
    'graph':        None,
    'optimizer':    optimizer.get_dotproduct_loss,
    'updater':      embedding.update_l2_embedding,
    'progress':     util.print_progress,
    'l2_reg':       0.01,
    'init_alpha':   0.025
}

current_update_times =  mp.RawValue('i', 0)
userEmbed =             None
itemEmbed =             None
######


### user functions ###
def create_graph(train_path, embedding_dimension=64, delimiter='\t'):
    global globalVariables
    global userEmbed
    global itemEmbed

    globalVariables['graph'] = graph.Graph(train_path, delimiter=delimiter, mode='edge')

    print('create embeddings...', end='', flush=True)
    userEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].vertex_count,
        dimensions=embedding_dimension)
    itemEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].context_count,
        dimensions=embedding_dimension)
    print('DONE', flush=True)

    return userEmbed, globalVariables['graph'].vertex_mapper, itemEmbed, globalVariables['graph'].context_mapper

def set_param(params):
    global globalVariables
    for key in params:
        globalVariables[key] = params[key]

def train(update_times=10, workers=1):
    global globalVariables
    globalVariables['total_update_times'] = int(update_times * 1000000)
    globalVariables['workers'] = workers
    globalVariables['worker_update_times'] = int((update_times * 1000000)/workers)
    globalVariables['min_alpha'] = globalVariables['init_alpha'] * 1000 / globalVariables['total_update_times']

    #util.optimize_numpy_multiprocessing(workers)

    processes = []
    for i in range(workers):
        p = mp.Process(target=learner, args=())
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    current_update_times.value = 0
    globalVariables['progress'](1.0)
    
def save_embeddings(file_prefix="mf"):
    global globalVariables
    global userEmbed
    global itemEmbed
    print()
    embedding.save_embeddings(userEmbed, globalVariables['graph'].vertices, file_prefix+'_vertex')
    embedding.save_embeddings(itemEmbed, globalVariables['graph'].contexts, file_prefix+'_context')
######


### main learner ###
def learner():
    globalVariables['graph'].cache_edge_samples(globalVariables['worker_update_times'])
    globalVariables['progress'](0.0)
    monitor_flag = int(1e3)
    _learning_rate = globalVariables['init_alpha']
    for i in range(1, globalVariables['worker_update_times']+1):
        user, user_idx, item, item_idx, weight = \
            globalVariables['graph'].draw_an_edge_from_sample()

        user_embedding = userEmbed[user_idx]
        item_embedding = itemEmbed[item_idx]

        user_loss, item_loss = globalVariables['optimizer'](user_embedding, item_embedding, weight)
        
        globalVariables['updater'](userEmbed, user_idx, user_loss, _learning_rate, globalVariables['l2_reg'])
        globalVariables['updater'](itemEmbed, item_idx, item_loss, _learning_rate, globalVariables['l2_reg'])

        if i % monitor_flag == 0:
            current_progress_percentage = current_update_times.value / globalVariables['total_update_times']
            _learning_rate = globalVariables['init_alpha'] * (1.0 - current_progress_percentage)
            _learning_rate = max(globalVariables['min_alpha'], _learning_rate)
            current_update_times.value += monitor_flag
            globalVariables['progress'](current_progress_percentage)
######


