from pysmore.libs import graph, optimizer, embedding, util
import multiprocessing as mp

### global variables ###
globalVariables = {
    'graph':        None,
    'optimizer':    optimizer.get_loglikelihood_loss,
    'updater':      embedding.update_l2_embedding,
    'progress':     util.print_progress,
    'init_alpha':   0.025,
    'l2_reg':       0.01,
    'num_negative': 3,
    'walk_length':  10,
    'window_size':  5
}

current_update_times =  mp.RawValue('i', 0)
vertexEmbed =           None
contextEmbed =          None
######


### user functions ###
def create_graph(train_path, embedding_dimension=64, delimiter='\t'):
    global globalVariables
    global vertexEmbed
    global contextEmbed

    globalVariables['graph'] = graph.Graph(train_path, delimiter=delimiter, mode='node', undirected=True)

    print('create embeddings...', end='', flush=True)
    vertexEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].vertex_count,
        dimensions=embedding_dimension)
    contextEmbed = embedding.create_embeddings_unsafe(
        amount=globalVariables['graph'].context_count,
        dimensions=embedding_dimension)
    print('DONE', flush=True)

    return vertexEmbed, globalVariables['graph'].vertex_mapper

def set_param(params):
    global globalVariables
    for key in params:
        globalVariables[key] = params[key]

def train(walk_times=10, workers=1):
    global globalVariables
    globalVariables['walk_times'] = walk_times
    globalVariables['total_update_times'] = walk_times * globalVariables['graph'].vertex_count
    globalVariables['workers'] = workers
    globalVariables['min_alpha'] = globalVariables['init_alpha'] * 1000 / globalVariables['total_update_times']

    #util.optimize_numpy_multiprocessing(workers)

    group_size_approax = globalVariables['graph'].vertex_count / workers 
    vertex_idxs_groups = [(int(i*group_size_approax), int((i+1)*group_size_approax)) for i in range(workers)]
    processes = []
    for i in range(workers):
        globalVariables['graph'].initialize_random_state() # otherwise each process uses the same random seed
        p = mp.Process(target=learner, args=(vertex_idxs_groups[i], ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    current_update_times.value = 0
    globalVariables['progress'](1.0)
    
def save_embeddings(file_prefix="deepwalk"):
    global globalVariables
    global vertexEmbed
    global contextEmbed
    print()
    embedding.save_embeddings(vertexEmbed, globalVariables['graph'].vertices, file_prefix+'_all')
######


### main learner ###
def learner(vertex_range):
    
    globalVariables['progress'](0.0)
    monitor_flag = int(1e2)
    _learning_rate = globalVariables['init_alpha']
    vertex_index_start, vertex_index_end = vertex_range

    for t in range(globalVariables['walk_times']):
        for i in range(vertex_index_start, vertex_index_end):
            start_vertex_idx = i
            start_vertex = globalVariables['graph'].vertices[start_vertex_idx]

            
            for vertex_idx, context_pos_idx in globalVariables['graph'].skipgram_generator(
                vertex=start_vertex,
                walk_length=globalVariables['walk_length'],
                window_size=globalVariables['window_size']):

                vertex_embedding = vertexEmbed[vertex_idx]
                context_pos_embedding = contextEmbed[context_pos_idx]
                
                vertex_loss, context_pos_loss = \
                    globalVariables['optimizer'](vertex_embedding, context_pos_embedding, 1.0)
            
                globalVariables['updater'](contextEmbed, context_pos_idx, context_pos_loss, _learning_rate, globalVariables['l2_reg'])

                context_neg, context_neg_idxs = \
                    globalVariables['graph'].draw_contexts_uniformly(amount=globalVariables['num_negative']) # should be negative not random
                for context_neg_idx in context_neg_idxs:
                    context_neg_embedding = contextEmbed[context_neg_idx]

                    vertex_neg_loss, context_neg_loss = \
                        globalVariables['optimizer'](vertex_embedding, context_neg_embedding, 0.0)
                    
                    globalVariables['updater'](contextEmbed, context_neg_idx, context_neg_loss, _learning_rate, globalVariables['l2_reg'])

                    vertex_loss += vertex_neg_loss

                globalVariables['updater'](vertexEmbed, vertex_idx, vertex_loss, _learning_rate, globalVariables['l2_reg'])
            

            if (i-vertex_index_start+1) % monitor_flag == 0:
                current_progress_percentage = current_update_times.value / globalVariables['total_update_times']
                _learning_rate = globalVariables['init_alpha'] * (1.0 - current_progress_percentage)
                _learning_rate = max(globalVariables['min_alpha'], _learning_rate)
                current_update_times.value += monitor_flag
                globalVariables['progress'](current_progress_percentage)
######