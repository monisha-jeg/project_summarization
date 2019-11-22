import operator
from queue import PriorityQueue
import torch as th


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        """
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(
    sents, graphs, encoder, decoder, vocab, device, max_len=2000, beam_width=10
):
    """
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    """
    import pdb

    pdb.set_trace()
    encoder_output = encoder(sents, graphs)
    decoder_input = th.tensor([vocab["<START>"]]).to(device)
    decoder_hidden = encoder_output[-1]
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []
    attentions = []

    # decoding goes sentence by sentence

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > max_len:
            break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == vocab["<END>"] and n.prevNode != None:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        decoder_output, decoder_hidden, attnetion = decoder(
            decoder_input, encoder_output, decoder_hidden
        )
        attentions.append(attnetion.cpu().numpy())

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = th.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[new_k].view(1, -1)
            log_p = log_prob[new_k].item()

            node = BeamSearchNode(
                decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1
            )
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)

        utterance = utterance[::-1]
        utterances.extend(utterance)

    decoded_batch = [vocab.idx_word[i.item()] for i in utterances]

    return decoded_batch[1:], attentions
