import torch
import json
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_json(filename):
    with open(filename) as f:
        vocab = json.load(f)
    return vocab


def save_json(filename, data):
    with open(filename, "w") as text_file:
        text_file.write(json.dumps(data))
    return
