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
import torch as th
from pysmore.model import mf

# select your device (optional)
# it depends on the pytorch version you installed
DEVICE = th.device("cpu")

# initialize graph
G = mf.Graph(device=DEVICE)

# create graph with DGL
G.create_graph("ui.train.txt", embedding_size=5)

# you can modify the DGLGraph by calling `G.graph`
# ex. add more NN layers to the graph, etc.

# start training!
# Noted that `update_times` will be multiplied by 1 million
G.train(update_times=1e-4, batch_size=1, neg_n=0)

# output the embeddings
G.save_embeddings("mf.rep")
```

## Task
Given a network input:
```txt
userA itemA 3
userA itemC 5
userB itemA 1
userB itemB 5
userC itemA 4
```
The model learns the representations of each vertex:
```
6 5
userA 0.0815412 0.0205459 0.288714 0.296497 0.394043
itemA -0.207083 -0.258583 0.233185 0.0959801 0.258183
itemC 0.0185886 0.138003 0.213609 0.276383 0.45732
userB -0.0137994 -0.227462 0.103224 -0.456051 0.389858
itemB -0.317921 -0.163652 0.103891 -0.449869 0.318225
userC -0.156576 -0.3505 0.213454 0.10476 0.259673
```
