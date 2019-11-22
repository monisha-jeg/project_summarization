from models.encoders import SentenceEncoder, ParagraphRNNEncoder, FullEncoder
from models.decoders import FinalDecoderRNN, RNNDecoder
from data_utils import data_to_graph
from preprocessing.process_dumps import Vocab
import random
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import rouge

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
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-TRAIN_DIR", default="xsum-dumps/train")
# TRAIN_DIR = "xsum-dumps/train"
parser.add_argument("-VAL_DIR", default="xsum-dumps/val")
# VAL_DIR = "xsum-dumps/val"
parser.add_argument("-WORD_DIM", default=300)
# WORD_DIM = 300
parser.add_argument("-WORD_GCN_HIDDEN", default=[300, 200, 200])
# WORD_GCN_HIDDEN = [300, 200, 200]
parser.add_argument("-WORD_GCN_OUT", default=200)
# WORD_GCN_OUT = 200
parser.add_argument("-GLOVE_PATH", default="glove.840B.300d.pkl")
# GLOVE_PATH = "glove.840B.300d.pkl"
parser.add_argument("-WORD_GCN_DROPOUT", default=0.0)
# WORD_GCN_DROPOUT = 0.0
parser.add_argument("-DECODER_DROPOUT", default=0.0)
# DECODER_DROPOUT = 0.0
parser.add_argument("-EPOCHS", default=2)
# EPOCHS = 2
parser.add_argument("-VOCAB_PATH", default="vocab_dump.pkl")
# VOCAB_PATH = "vocab_dump.pkl"
parser.add_argument("-USE_CUDA", default=1)
# USE_CUDA = True
parser.add_argument("-PARA_RNN_LAYERS", default=2)
# PARA_RNN_LAYERS = 2
parser.add_argument("-PARA_RNN_DROPOUT", default=0.0)
# PARA_RNN_DROPOUT = 0.0
parser.add_argument("-PARA_RNN_OUTDIM", default=200)
# PARA_RNN_OUTDIM = 100
parser.add_argument("-LR", default=0.001)
parser.add_argument("-MAX_LEN", default=100)
parser.add_argument("-MODEL_DIR", default="WORDGCN-RNN")
parser.add_argument("-SAVE_EVERY", default=100)
parser.add_argument("-EVAL_EVERY", default=100)

args = parser.parse_args()
os.makedirs(args.MODEL_DIR, exist_ok=True)
writer = SummaryWriter(log_dir=args.MODEL_DIR, flush_secs=30)


# TODO: Integrate process_dump.py to train.py
if args.USE_CUDA:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
else:
    device = th.device("cpu")

train_files = [
    os.path.join(args.TRAIN_DIR, f)
    for f in os.listdir(args.TRAIN_DIR)
    if os.path.isfile(os.path.join(args.TRAIN_DIR, f))
]
val_files = [
    os.path.join(args.VAL_DIR, f)
    for f in os.listdir(args.VAL_DIR)
    if os.path.isfile(os.path.join(args.VAL_DIR, f))
]


with open(args.VOCAB_PATH, "rb") as fl:
    vocab = pl.load(fl)

with open(args.GLOVE_PATH, "rb") as fl:
    emb_weights = pl.load(fl)
emb_weights = th.FloatTensor(emb_weights).to(device)


decoder = RNNDecoder(
    vocab_size=vocab.word_count,
    word_embedding=args.WORD_DIM,
    weight_matrix=emb_weights,
    dropout=args.DECODER_DROPOUT,
    num_layers=1,
    enc_dim=args.PARA_RNN_OUTDIM,
).to(device)


def get_data(file_path):
    data, summ = data_to_graph(file_path, vocab)
    sents = []
    graphs = []
    for sent, graph in data:
        sent = th.LongTensor(sent).to(device)
        sents.append(sent)
        graph = DGLGraph(graph)
        graph.add_edges(graph.nodes(), graph.nodes())
        graphs.append(graph)
    summ = th.LongTensor(summ).to(device)
    return sents, graphs, summ


fullencoder = FullEncoder(
    args, word_count=vocab.word_count, emb_weights=emb_weights
).to(device)

for p in fullencoder.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# sents, graphs, summ = get_data(train_files[0])
# out_rnn = fullencoder(sents, graphs)


teacher_forcing_ratio = 0.5


def train(
    sents,
    graphs,
    summary_tensor,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    criterion,
    max_length=args.MAX_LEN,
):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    target_length = summary_tensor.size()[0]
    loss = 0
    encoder_outputs = encoder(sents, graphs)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_input = summary_tensor[0]
    decoder_hidden = encoder_outputs[-1]
    decoder_outputs = []
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(1, target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, encoder_outputs, decoder_hidden
            )
            # loss += criterion(decoder_output, summary_tensor[di])
            decoder_input = summary_tensor[di]  # Teacher forcing
            decoder_outputs.append(decoder_output)

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(1, target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, encoder_outputs, decoder_hidden
            )
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            # loss += criterion(decoder_output, summary_tensor[di])
            decoder_outputs.append(decoder_output)
            if decoder_input.item() == vocab["<END>"]:
                break
    decoder_outputs = th.stack(decoder_outputs)
    loss = criterion(decoder_outputs, summary_tensor[1 : di + 1])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def evaluate(sents, graphs, encoder, decoder, max_length=args.MAX_LEN, vocab=vocab):
    encoder_outputs = encoder(sents, graphs)
    decoder_input = th.tensor([vocab["<START>"]]).to(device)
    decoder_hidden = encoder_outputs[-1]
    decoder_outputs = []
    for di in range(1, max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, encoder_outputs, decoder_hidden
        )
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        # loss += criterion(decoder_output, summary_tensor[di])
        decoder_outputs.append(decoder_input.item())
        if decoder_input.item() == vocab["<END>"]:
            break
    decoder_words = [vocab.idx_word[i] for i in decoder_outputs]
    return decoder_words, decoder_attention[: di + 1]


def get_performance(val_files, encoder, decoder, vocab=vocab, dest_file=None):
    references = []
    predicted = []
    encoder.eval()
    decoder.eval()
    for f in tqdm(val_files):
        with th.no_grad():
            sents, graphs, summ = get_data(f)
            if len(sents) > 0:
                references.append([vocab.idx_word[i] for i in summ[1:]])
                pred, _ = evaluate(sents, graphs, encoder, decoder, vocab=vocab)
                predicted.append(pred)
    bleu = np.mean([sentence_bleu([r], p) for r, p in zip(references, predicted)])
    evaluator = rouge.Rouge(
        metrics=["rouge-n", "rouge-l", "rouge-w"],
        max_n=4,
        limit_length=True,
        length_limit=100,
        length_limit_type="words",
        # apply_avg=apply_avg,
        apply_best=True,
        alpha=0.5,  # Default F1_score
        weight_factor=1.2,
        stemming=True,
    )
    rouge_scores = evaluator.get_scores(
        [" ".join(p) for p in predicted], [" ".join(r) for r in references]
    )
    if dest_file is not None:
        with open(dest_file, "w") as fl:
            for r, p in zip(references, predicted):
                fl.write(" ".join(r) + "\n")
                fl.write(" ".join(p) + "\n\n")

    return bleu, rouge_scores


iters = 0


def trainIters():
    global iters
    encoder_optimizer = th.optim.Adam(fullencoder.parameters(), lr=args.LR)
    decoder_optimizer = th.optim.Adam(decoder.parameters(), lr=args.LR)
    fullencoder.train()
    decoder.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.EPOCHS):
        print("Epoch %d" % epoch)
        l = 1.1
        for f in tqdm(train_files):
            iters += 1
            sents, graphs, summ = get_data(f)
            if len(sents) > 0:
                l = train(
                    sents,
                    graphs,
                    summ,
                    fullencoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                )
            th.cuda.empty_cache()

            writer.add_scalar("Loss", l, iters)
            if iters % args.SAVE_EVERY == 0:
                th.save(
                    {
                        "epoch": epoch,
                        "encoder_state_dict": fullencoder.state_dict(),
                        "decoder_state_dict": decoder.state_dict(),
                        "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                        "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                        "loss": l,
                    },
                    os.path.join(args.MODEL_DIR, "model%d.pth" % (iters)),
                )
            if iters % args.EVAL_EVERY == 0:
                scores = get_performance(
                    random.sample(val_files, 1000), fullencoder, decoder
                )
                fullencoder.train()
                decoder.train()
                writer.add_scalar("BLUE", scores[0], iters)
                for k in scores[1].keys():
                    writer.add_scalar(k, scores[1][k]["f"], iters)

