# pySmore

- DEVELOPING...
- Original C++ version: [SMORe](https://github.com/cnclabs/smore)

## Feature checklist:
- [x] MF (**M**atrix **F**actorization)
- [x] BPR (**B**ayesian **P**ersonalized **R**anking)
  - [BPR: Bayesian personalized ranking from implicit feedback](https://dl.acm.org/citation.cfm?id=1795167)
- [ ] LINE(**L**arge-scale **I**nformation **N**etwork **E**mbedding)
  - [LINE: Large-scale Information Network Embedding](http://dl.acm.org/citation.cfm?id=2741093)
- [ ] DeepWalk
  - [DeepWalk: online learning of social representations](http://dl.acm.org/citation.cfm?id=2623732)
- [ ] Walklets
  - [Don't Walk, Skip! Online Learning of Multi-scale Network Embeddings](https://arxiv.org/abs/1605.02115) 
- [ ] HPE (**H**eterogeneous **P**reference **E**mbedding)
  - [Query-based Music Recommendations via Preference Embedding](http://dl.acm.org/citation.cfm?id=2959169)
- [ ] APP (**A**symmetric **P**roximity **P**reserving graph embedding)
  - [Scalable Graph Embedding for Asymmetric Proximity](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14696)
- [ ] WARP-like
  - [WSABIE: Scaling Up To Large Vocabulary Image Annotation](https://dl.acm.org/citation.cfm?id=2283856)
  - [Learning to Rank Recommendations with the k-Order Statistic Loss](https://dl.acm.org/citation.cfm?id=2507157.2507210)
- [ ] HOP-REC
  - [HOP-Rec: High-Order Proximity for Implicit Recommendation](https://dl.acm.org/citation.cfm?id=3240381)
- [ ] CSE (named nemf & nerank in cli)
  - [Collaborative Similarity Embedding for Recommender Systems](https://arxiv.org/abs/1902.06188)

## Run Example
```cmd
python3 example.py
```

## Get started
```python
from pysmore import models

# Choose a graph embedding method
trainer = models.MF

# Create a graph with given user-item interaction data
trainer.create_graph("data/ui.train.txt", embedding_dimension=6)

# Pass the parameters that you'd like for training!
trainer.set_param({
    'init_lr':  0.025,  # initial learning rate
    'l2_reg':   0.01    # L2-Regularize ratio
})

# Start training!
# Noted that `update_times` will be multiplied by 1 million
trainer.train(update_times=1e-4, workers=4)

# Afterwards, output the embeddings.
trainer.save_embeddings(file_prefix="mf")
```

## Task
Given a network input (split by `\t`):
```txt
userA	itemA	3
userA	itemC	5
userB	itemA	1
userB	itemB	5
userC	itemA	4
```
The model learns the representations of each vertex:
```
userA 0.45947249 0.50407268 -0.48727296 0.17812133 -0.12710278 -0.32385066
itemA 0.43970733 0.50159092 -0.47445659 0.17394969 -0.10405107 -0.30718780
itemC -0.06104707 -0.02211397 0.03871192 0.01201094 0.18371690 0.01444941
userB -0.51468509 0.25420145 0.08170974 0.07961795 0.39588787 0.42408562
itemB -0.57684131 0.28458484 0.09723139 0.10464700 0.51066395 0.46714100
userC -0.31679640 0.17126887 0.04893538 0.08118480 0.38689152 0.24092283
```
