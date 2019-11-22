import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from models.encoders import WordEmbedder

MAX_LENGTH = 20


class FinalDecoderRNN(nn.Module):
    def __init__(self, word_dim, vocab_size, dropout_rate=0.1, max_length=MAX_LENGTH):
        super(FinalDecoderRNN, self).__init__()
        self.hidden_size = word_dim
        self.output_size = vocab_size
        self.dropout_p = dropout_rate
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(th.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = th.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = th.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights


class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        word_embedding=300,
        weight_matrix=None,
        dropout=0.0,
        num_layers=1,
        hidden_size=200,
        bidirectional=True,
        atten_dim=100,
        enc_dim=200,
    ):
        super(RNNDecoder, self).__init__()
        self.word_emb = WordEmbedder(vocab_size, word_embedding, weight_matrix)
        self.hidden_size = hidden_size
        self.atten_W1 = nn.Linear(hidden_size, atten_dim)
        self.atten_W2 = nn.Linear(enc_dim, atten_dim)
        self.atten_v = nn.Parameter(th.FloatTensor(atten_dim))
        self.num_layers = num_layers
        self.lin = nn.Linear(word_embedding + enc_dim, hidden_size)
        self.rnn_decoder = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.pred_word = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_idx, sents, hidden=None):
        embs = self.word_emb(input_idx).squeeze()
        attentions = F.softmax(
            F.tanh(self.atten_W1(hidden) + self.atten_W2(sents)) @ self.atten_v, -1
        )
        weight_enc = attentions @ sents
        output = F.relu(self.lin(th.cat((embs, weight_enc))))
        output, hidden = self.rnn_decoder(
            output.view(1, 1, -1), hidden.view(self.num_layers, 1, -1)
        )
        output = self.pred_word(output)
        return output.squeeze(), hidden.squeeze(), attentions

