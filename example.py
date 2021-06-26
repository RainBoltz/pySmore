import pysmore.models.mf as MF
# also try out BPR with:
# `import pysmore.models.bpr as BPR`

# Choose a graph embedding method
trainer = MF

# Create a graph with given data
trainer.create_graph("data/ui.train.txt", embedding_dimension=6)

# Pass the parameters that you'd like for training!
trainer.set_param({
    'init_lr':  0.025,  # initial learning rate
    'l2_reg':   0.01    # L2-Regularize ratio
})

# Start training!
# Noted that `update_times` will be multiplied by 1 million
trainer.train(update_times=1e-4, workers=4)

# output the embeddings
# Note that it will generate four files:
# 1. <file_prefix>_user.idx:  user names
# 2. <file_prefix>_user.rep:  user embeddings
# 3. <file_prefix>_item.idx:  item names
# 4. <file_prefix>_item.rep:  item embeddings
trainer.save_embeddings(file_prefix="mf")
