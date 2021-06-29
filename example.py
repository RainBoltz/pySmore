import pysmore.models.bpr as BPR
# also try out LINE with:
# `import pysmore.models.line as LINE`

# Choose a graph embedding method
trainer = BPR

# Create a graph with given data
trainer.create_graph("data/ui.train.txt", embedding_dimension=6)

# Pass the parameters that you'd like for training!
trainer.set_param({
    'init_lr':      0.025,  # initial learning rate
    'l2_reg':       0.01,   # L2-Regularize ratio
    'num_negative': 1       # negative sampling amount
})

# Start training!
# Noted that `update_times` will be multiplied by 1 million
trainer.train(update_times=0.05, workers=2)

# output the embeddings
# Note that it will generate files with given `file_prefix`,
# e.i. "bpr_vertex.rep", "bpr_vertex.idx", ...
trainer.save_embeddings(file_prefix="bpr")
