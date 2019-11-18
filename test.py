from preprocessing.process_dumps import Vocab
import numpy as np
from six.moves import cPickle as pl
from models.encoders import SentenceEncoder, WordEmbedder, GCNNet, SumPooler

with open('vocab_dump.pkl', 'rb') as fl:
    vocab = pl.load(fl)

with open('glove.840B.300d.pkl', 'rb') as fl:
    emb_weights = pl.load(fl)

