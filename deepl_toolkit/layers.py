import torch
from torch import nn
import numpy as np
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class attention_layer(nn.Module):
    """
    Attention Layer

    """

    def __init__(self, input_dim, hidden_dim):
        super(attention_layer, self).__init__()
        self.Wv = nn.Conv2d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.Wu = nn.Linear(input_dim, hidden_dim)
        self.Wp = nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0)

        self.hidden_dim = hidden_dim
        self.attention_maps = None

    def getMap(self):
        """
        Get saved attention map

        return: Attention map

        """

        return torch.squeeze(self.attention_maps[0], 1)

    def forward(self, v, u):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Forward Pass

        v: visual features (N x D x H x W)
        u: attention key (N x D)
        return: attention map (N x D)

        """

        N, K = v.size(0), self.hidden_dim
        D, H, W = v.size(1), v.size(2), v.size(3)
        v_proj = self.Wv(v)  # N x K x H x W
        u_proj = self.Wu(u)  # N x K
        u_proj_expand = u_proj.view(N, K, 1, 1).expand(N, K, H, W)
        h = F.tanh(v_proj + u_proj_expand)
        h = self.Wp(h).view(N, H * W)
        p = F.softmax(h, -1).view(N, 1, H, W)
        self.attention_maps = p.data.clone()

        v_tilde = (p.expand_as(v) * v).sum(3).sum(2).view(N, D)
        return v_tilde



class seq2seq_rnn(nn.Module):
    def __init__(self, input_features, rnn_features, num_layers=1, drop=0.0,
               rnn_type='LSTM', rnn_bidirectional=False):
        super(seq2seq_rnn, self).__init__()
        self.bidirectional = rnn_bidirectional

        if rnn_type == 'LSTM':
          self.rnn = nn.LSTM(input_size=input_features,
                    hidden_size=rnn_features, dropout=drop,
                    num_layers=num_layers, batch_first=True,
                    bidirectional=rnn_bidirectional)
        elif rnn_type == 'GRU':
          self.rnn = nn.GRU(input_size=input_features,
                    hidden_size=rnn_features, dropout=drop,
                    num_layers=num_layers, batch_first=True,
                    bidirectional=rnn_bidirectional)
        else:
          raise ValueError('Unsupported Type')

        self.init_weight(rnn_bidirectional, rnn_type)

    def init_weight(self, bidirectional, rnn_type):
        self._init_rnn(self.rnn.weight_ih_l0, rnn_type)
        self._init_rnn(self.rnn.weight_hh_l0, rnn_type)
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.bias_hh_l0.data.zero_()

        if bidirectional:
          self._init_rnn(self.rnn.weight_ih_l0_reverse, rnn_type)
          self._init_rnn(self.rnn.weight_hh_l0_reverse, rnn_type)
          self.rnn.bias_ih_l0_reverse.data.zero_()
          self.rnn.bias_hh_l0_reverse.data.zero_()

    def _init_rnn(self, weight, rnn_type):
        chunk_size = 4 if rnn_type == 'LSTM' else 3
        for w in weight.chunk(chunk_size, 0):
            init.xavier_uniform(w)

    def forward(self, q_emb, lengths):
        lens, indices = torch.sort(lengths, 0, True)

        packed = pack_padded_sequence(q_emb[indices.to(device)], lens.tolist(), batch_first=True)
        if isinstance(self.rnn, nn.LSTM):
            _, ( outputs, _ ) = self.rnn(packed)
        elif isinstance(self.rnn, nn.GRU):
            _, outputs = self.rnn(packed)

        if self.bidirectional:
            outputs = torch.cat([ outputs[0, :, :], outputs[1, :, :] ], dim=1)
        else:
            outputs = outputs.squeeze(0)

        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices.to(device)]

        return outputs


class glove_embedding(nn.Module):
    """
    Embedding layer pre-trained with GloVE

    """

    def __init__(self, glove_vectors, dict, embed_dim=100, freezed=False):
        super(glove_embedding, self).__init__()
        self.word_to_idx = {}
        self.idx_to_word = dict

        weights_matrix = self.init_vectors(glove_vectors, embed_dim)
        num_embeddings, embedding_dim = weights_matrix.shape

        self.emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        self.emb_layer.weight = nn.Parameter(torch.from_numpy(weights_matrix))

        if freezed:
            self.emb_layer.weight.requires_grad = False

        del glove_vectors
        del weights_matrix

    def init_vectors(self, glove_vectors, embed_dim=100):
        weights_matrix = np.zeros((len(self.idx_to_word), embed_dim))
        for i, word in enumerate(self.idx_to_word):
            self.word_to_idx[word] = i
            try:
                weights_matrix[i] = glove_vectors[word]
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embed_dim,))
        return weights_matrix

    def forward(self, x):
        return self.emb_layer(x)


class gated_tanh(nn.Module):
    """
    Gated tanh activation function

    """

    def __init__(self, input_dim, output_dim):
        super(gated_tanh, self).__init__()
        self.f1 = nn.Linear(input_dim, output_dim)
        self.f2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_tilde = nn.functional.tanh(self.f1(x))
        g = nn.functional.sigmoid(self.f2(x))
        return y_tilde*g


class contrastive_loss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 0 if samples are from the same class and label == 1 otherwise

    """

    def __init__(self, margin, temperature=1):
        super(contrastive_loss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.criterion = nn.MSELoss(reduction='none').to(device)
        self.temperature = temperature

    def forward(self, output1, output2, target):

        euclidean_distance = self.criterion(output1, output2).mean(1)
        loss_contrastive = torch.mean((1 - target) * euclidean_distance +
                                      (target) * torch.clamp(self.margin - euclidean_distance, min=0.0))

        return loss_contrastive / self.temperature
