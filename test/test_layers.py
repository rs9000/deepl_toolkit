import torch
import deepl_toolkit
from gensim.models import KeyedVectors

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_bi_lstm_encoder():
    glove_path, dict_path = "/nas/softechict-nas-1/rdicarlo/GLOVE/glove.6B.100d.vec", \
                            "/nas/softechict-nas-1/rdicarlo/dict/dict.json"
    glove_vectors = KeyedVectors.load_word2vec_format(glove_path)
    dict = deepl_toolkit.utils.load_json(dict_path)
    emb_layer = deepl_toolkit.layers.glove_embedding(glove_vectors, dict).to(device)
    rnn_layer = deepl_toolkit.layers.bi_lstm_encoder(emb_layer,200)

    x = torch.LongTensor([4,12,3,44,21,43,98,2,1,1,1]).to(device)
    x = x.expand(10, -1)
    x_encoded = rnn_layer(x)
    return x_encoded


def test_glove_embedding():
    glove_path, dict_path = "/nas/softechict-nas-1/rdicarlo/GLOVE/glove.6B.100d.vec", \
                            "/nas/softechict-nas-1/rdicarlo/dict/dict.json"
    glove_vectors = KeyedVectors.load_word2vec_format(glove_path)
    dict = deepl_toolkit.utils.load_json(dict_path)
    emb_layer = deepl_toolkit.layers.glove_embedding(glove_vectors, dict).to(device)

    x = torch.LongTensor([4,12,3,44,21,43,98,2,1,1,1]).to(device)
    x = x.expand(10, -1)
    x_embed = emb_layer(x)
    return x_embed


if __name__ == '__main__':
    # Entry point
    test_bi_lstm_encoder()
    test_glove_embedding()
