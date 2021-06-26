import numpy as np

U = np.array([  [1, 2, 3, 4, 5],
                [3, 4, 4, 3, 4],
                [5, 6, 4, 7, 1]])

uids_range = U.shape[0]

I = np.random.uniform(low=-0.5, high=0.5, size=(7,5))/5

iids_range = I.shape[0]

Sampler = np.random.default_rng()


uid = Sampler.choice(uids_range, 1, replace=False)
iids = Sampler.choice(iids_range, 1, replace=False)[0]
iids2 = Sampler.choice(iids_range, 5, replace=False)

from_embedding = U[uid]
to_embedding_pos = I[iids]
to_embedding_negs = I[iids2]

diff_to_embedding = to_embedding_pos - to_embedding_negs
prediction = np.dot(from_embedding, diff_to_embedding.T)
gradient = np.apply_along_axis(lambda x: -x, axis=0, arr=prediction)

print(diff_to_embedding)
print(prediction)
print(gradient)


