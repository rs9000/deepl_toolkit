import torch
from deepl_toolkit import vectors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    # Entry point
    a = torch.rand(10, 80).to(device)
    b = torch.rand(10, 32).to(device)
    u = torch.matmul(a.unsqueeze(2), b.unsqueeze(1))
    u = vectors.random_projection(u, 25)
