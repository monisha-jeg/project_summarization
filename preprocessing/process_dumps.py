import os 
try:
	import cPickle as pl
except ImportError:
	from six.moves import cPickle as pl
from tqdm import tqdm
import numpy as np



class Vocab(object):
    def __init__(self, word_transform = str.lower, reverse_transform = str):
        self.word_idx = {"<START>" : 0, "<END>" : 1, "<PAD>" : 2, "<UNK>" : 3}
        self.idx_word = ["<START>","<END>" , "<PAD>", "<UNK>"]
        self.word_count = 4
        self.transformer = word_transform
        self.reverse_transform = reverse_transform
    
    def add_word(self, word):
        word = self.transformer(word)
        if word not in self.word_idx.keys():
            self.word_idx[word] = self.word_count
            self.idx_word.append(word)
            self.word_count += 1
        
    def add_list(self, word_list):
        for w in word_list:
            self.add_word(w)
    
    
    
    def __getitem__(self, word):
        if word not in ["<START>","<END>" , "<PAD>", "<UNK>"]:
            word = self.transformer(word)
        return self.word_idx[word] if word in self.word_idx.keys() else self.word_idx["<UNK>"]
    
    def get_word(self, idx):
        if idx < self.word_count:
            return self.reverse_transform(self.idx_word[idx])
        return "<UNK>"
    


if __name__ == "__main__":
    DIRS = ['xsum-dumps/train', 'xsum-dumps/test']
    dump_file = "vocab_dump.pkl"
    MIN_FREQ = 5
    v = Vocab()
    word_freqs = {}
    for d in DIRS:
        for f in tqdm(os.listdir(d)):
            with open(os.path.join(d,f), 'rb') as fl:
                data = pl.load(fl)
            for sent in data['text']:
                for w in sent:
                    if w.lower() in word_freqs.keys():
                        word_freqs[w.lower()] += 1
                    else:
                        word_freqs[w.lower()] = 1
            for w in data['summary']:
                if w.lower() in word_freqs.keys():
                    word_freqs[w.lower()] += 1
                else:
                    word_freqs[w.lower()] = 1
    to_del = []
    for word in word_freqs.keys():
        if word_freqs[word] > MIN_FREQ:
            v.add_word(word)
    
    

    print("Done building Vocabulary!")
    print("Total number of words: %d"%(v.word_count))

    with open(dump_file, 'wb') as fl:
        pl.dump(v, fl)

    embeds = {}
    embed_file = 'glove.840B.300d.txt'
    for line in tqdm(open(embed_file, 'r'), total=2196017):
        l = line.split()
        if v.transformer(l[0]) in v.word_idx.keys():
            embeds[v.transformer(l[0])] = np.array(l[-300:], dtype='float')

    embed_matrix = [np.random.rand(300) for _ in range(4)]
    ct = 0
    for i in tqdm(range(4, v.word_count)):
        if v.idx_word[i] in embeds.keys():
            embed_matrix.append(embeds[v.idx_word[i]])
        else:
            embed_matrix.append(np.random.rand(300))
            ct+= 1

    embed_matrix = np.array(embed_matrix)
    print("Shape of embed weight matrix:", embed_matrix.shape)
    with open(embed_file.replace('txt','pkl'), 'wb') as fl:
        pl.dump(embed_matrix, fl)
