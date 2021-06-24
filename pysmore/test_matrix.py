import numpy as np

U = np.array([  [1, 2, 3, 4, 5],
                [3, 4, 4, 3, 4],
                [5, 6, 4, 7, 1]])

uids_range = U.shape[0]

I = np.random.uniform(low=-0.5, high=0.5, size=(7,5))

iids_range = I.shape[0]

Sampler = np.random.default_rng()


uid = Sampler.choice(uids_range, 1, replace=False)
iids = Sampler.choice(iids_range, 1, replace=False)
iids2 = Sampler.choice(iids_range, 5, replace=False)

print(I)
print(iids2)
I[iids2] -= I[iids2]
print(I)



