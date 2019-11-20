import numpy as np
import torch as th
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from six.moves import cPickle as pl
import networkx as nx


def data_to_graph(data, vocab_obj):
    if type(data) is str:
        data = pl.load(open(data, "rb"))
    idx_sent = [
        np.array(
            [vocab_obj["<START>"]]
            + [vocab_obj[w] for w in sent]
            + [vocab_obj["<END>"]],
            dtype="int",
        )
        for sent in data["text"]
    ]
    dep_graphs = [nx.DiGraph() for _ in range(len(idx_sent))]
    for g, deps, idx in zip(dep_graphs, data["dep"], idx_sent):
        g.add_nodes_from(list(range(len(idx))))
        g.add_edges_from(deps)
    return [(i, d) for i, d in zip(idx_sent, dep_graphs)]
