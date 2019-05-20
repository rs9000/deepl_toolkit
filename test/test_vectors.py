import torch
import deepl_toolkit
from gensim.models import KeyedVectors

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_random_projection():
    a = torch.rand(10, 80).to(device)
    b = torch.rand(10, 32).to(device)
    u = torch.matmul(a.unsqueeze(2), b.unsqueeze(1))
    u = deepl_toolkit.vectors.random_projection(u, 25)


if __name__ == '__main__':
    # Entry point
    test_random_projection()
