import torch
from torch import nn
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class bi_lstm_encoder(nn.Module):
    """
    LSTM encoder
    Embedding(question) -> RNN()

    """

    def __init__(self, embedding_layer, rnn_dim, embedding_size=100, rnn_num_layers=2, rnn_dropout=0.4):
        super(bi_lstm_encoder, self).__init__()

        self.NULL = 1
        self.START = 0
        self.END = 2

        self.embed = embedding_layer
        self.rnn_size = rnn_dim
        self.rnn = nn.LSTM(embedding_size, rnn_dim, rnn_num_layers,
                           dropout=rnn_dropout, batch_first=True, bidirectional=True).to(device)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in RNN

        """

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param.data)
            if 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        # type: (torch.Tensor) -> (torch.Tensor, torch.Tensor)
        """
        Forward Pass

        x: question batch_size x question_len

        return: hs = question (batch_size X question_len X rnn_dim)
                idx = question-length, without padding (batch_size x 1)

        """

        N, T = x.shape
        idx = torch.LongTensor(N).fill_(T - 1)

        # Find the last non-null element in each sequence
        x_cpu = x.data.cpu()
        for i in range(N):
            for t in range(T - 1):
                if x_cpu[i, t] != self.NULL and x_cpu[i, t + 1] == self.NULL:
                    idx[i] = t
                    break
        idx = idx.type_as(x.data).long()
        idx.requires_grad = False

        hs, _ = self.rnn(self.embed(x))

        hs = hs.view(N, T, self.rnn_size, 2)

        # Split in forward and backward sequence
        output_forward, output_backward = torch.chunk(hs, 2, 3)
        output_forward = output_forward.squeeze(3)  # (batch_size, T, hidden_size)
        output_backward = output_backward.squeeze(3)  # (batch_size, T, hidden_size)

        # Find last elements of the forward sequence
        q_len = idx.view(N, 1, 1).expand(N, 1, self.rnn_size).to(device)

        # Trunk the forward sequence at t = question_len
        output_forward = output_forward.gather(1, q_len).view(N, self.rnn_size)
        # Get last state of the backward sequence t = 0
        output_backward = output_backward[:, 0, :]
        # Re-concat output
        output = torch.cat((output_forward, output_backward), -1)
        return output


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

    def __init__(self, margin):
        super(contrastive_loss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.criterion = nn.MSELoss(reduction='none').cuda()

    def forward(self, output1, output2, target, magnitude=1):

        euclidean_distance = self.criterion(output1, output2).mean(1)
        loss_contrastive = torch.mean((1 - target) * euclidean_distance +
                                      (target) * torch.clamp(self.margin - euclidean_distance, min=0.0)).mul(magnitude)

        return loss_contrastive
