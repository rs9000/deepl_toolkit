import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def random_projection(x, dim): 

    values = [1, -1, 0, 0, 0, 0] 
    r = torch.zeros(dim, x.size(-2))

    for i in range(0, r.size(0)): 
        for j in range(0, r.size(1)): 
            r[i, j] = random.sample(values, 1)[0] * (3**(1/2)) 

    r = r.expand(x.shape[0], -1, -1).to(device)
    out = torch.matmul(r, x)
    return out
