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
cd pySmore
python3 example.py
```

## Get started
```python
import pysmore.models.mf as MF
import pysmore.models.bpr as BPR

# Choose a graph embedding method
trainer = MF  # or BPR

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
```txt
userA	-0.068195	0.105852	0.056242	0.084970	-0.209601	-0.018169
itemA	-0.033628	0.046754	0.030732	0.035540	-0.105440	-0.008107
itemC	-0.114769	0.181540	0.092762	0.146976	-0.351410	-0.031107
userB	-0.050903	0.013206	0.077547	-0.013286	-0.179966	-0.003265
itemB	-0.088020	0.015789	0.137878	-0.031181	-0.313660	-0.004543
userC	-0.060036	0.086218	0.053508	0.066621	-0.187579	-0.014900
```
We can calculate the results (dot product): 
```txt
userA	itemA	0.034238
userA	itemC	0.118970
userB	itemA	0.023242
userB	itemB	0.072258
userC	itemA	0.029961
```