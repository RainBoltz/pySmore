from pysmore.libs import graph, optimizer, embedding, util
import multiprocessing as mp

# main learner
def learner():
    globalVariables['graph'].cache_edge_samples(globalVariables['update_times'])
    globalVariables['progress'](0.0)
    for i in range(1, globalVariables['update_times']+1):
        user, user_idx, item, item_idx, weight = \
            globalVariables['graph'].draw_an_edge_from_sample()

        user_embedding = userEmbed[user_idx]
        item_embedding = itemEmbed[item_idx]

        user_loss, item_loss = globalVariables['optimizer'](user_embedding, item_embedding, weight)
        
        current_progress_percentage = current_update_times.value / globalVariables['total_update_times']
        learning_rate = globalVariables['init_lr'] * (1.0 - current_progress_percentage)
        learning_rate = max(globalVariables['min_lr'], learning_rate)
        globalVariables['updater'](userEmbed, user_idx, user_loss, learning_rate, globalVariables['l2_reg'])
        globalVariables['updater'](itemEmbed, item_idx, item_loss, learning_rate, globalVariables['l2_reg'])

        monitor_flag = int(1e3)
        if i % monitor_flag == 0:
            current_update_times.value += monitor_flag
            globalVariables['progress'](current_progress_percentage)



# settings
train_path = r'/tmp2/ccwang/Disserstation/dataset/single/25/Digital_Music.train.tsv'
save_name = 'mf'
update_times = 10
init_learning_rate = 0.025
minimal_learning_rate = init_learning_rate * 1e-4
l2_regularize_ratio = 1e-2
workers = 20

total_update_times = int(update_times * 1e6)
util.optimize_numpy_multiprocessing(workers)

# global constants
globalVariables = {
    'graph':     graph.Graph(train_path, mode='edge'),
    'optimizer': optimizer.get_dotproduct_loss,
    'updater':   embedding.update_l2_embedding,
    'progress':  util.print_progress,
    'total_update_times': total_update_times,
    'init_lr': init_learning_rate,
    'update_times': int(total_update_times / workers),
    'min_lr': minimal_learning_rate,
    'l2_reg': l2_regularize_ratio
}

#create shared memory variables
print('create embeddings...', end='', flush=True)
userEmbed = embedding.create_embeddings_unsafe(amount=globalVariables['graph'].user_count)
itemEmbed = embedding.create_embeddings_unsafe(amount=globalVariables['graph'].item_count)
print('DONE', flush=True)

current_update_times = mp.RawValue('i', 0)

# main process
processes = []
for i in range(workers):
    p = mp.Process(target=learner, args=())
    p.start()
    processes.append(p)
for p in processes:
    p.join()
globalVariables['progress'](1.0)

# output
print()
embedding.save_embeddings(userEmbed, globalVariables['graph'].users, save_name+'_users')
embedding.save_embeddings(itemEmbed, globalVariables['graph'].items, save_name+'_items')
