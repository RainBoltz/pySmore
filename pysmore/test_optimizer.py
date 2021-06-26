from libs.optimizer import get_dotproduct_loss
import numpy as np

a = np.array([0.1,0.2,0.3])
b = np.array([0,0.2,0])

L = get_dotproduct_loss(a, b, 1.0)

print(L)