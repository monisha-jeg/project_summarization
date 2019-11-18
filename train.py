from models.encoders import SentenceEncoder
from models.decoders import FinalDecoderRNN
from data_utils import data_to_graph
from preprocessing.process_dumps import Vocab

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from dgl import DGLGraph

from six.moves import cPickle as pl
import os
from tqdm import tqdm

TRAIN_DIR = 'xsum-dumps/train'
VAL_DIR = 'xsum-dumps/val'
WORD_DIM = 300
WORD_GCN_HIDDEN = [300,200,200]
WORD_GCN_OUT = 200
GLOVE_PATH = 'glove.840B.300d.pkl'
WORD_GCN_DROPOUT = 0.11
EPOCHS = 2
VOCAB_PATH = 'vocab_dump.pkl'
USE_CUDA = True
# TODO: Integrate process_dump.py to train.py
if USE_CUDA:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
else:
    device = th.device("cpu")

train_files = [os.path.join(TRAIN_DIR, f) 
                    for f in os.listdir(TRAIN_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, f))]
val_files = [os.path.join(VAL_DIR, f)
                    for f in os.listdir(VAL_DIR) if os.path.isfile(os.path.join(TRAIN_DIR, f))]


with open(VOCAB_PATH, 'rb') as fl:
    vocab = pl.load(fl)

with open(GLOVE_PATH, 'rb') as fl:
    emb_weights = pl.load(fl)
emb_weights = th.FloatTensor(emb_weights).to(device)

encoder = SentenceEncoder(vocab_size = vocab.word_count, word_dim = WORD_DIM, out_word_dim = WORD_GCN_OUT,
                        word_gcn_hidden = WORD_GCN_HIDDEN, weight_matrix = emb_weights,
                        word_gcn_dropout = WORD_GCN_DROPOUT)

