import cPickle as pl 

data = pl.load(open("dump/train_binarized.pkl", 'rb'))
print "Data length", len(data)
obj = data[0]
print "First instance has " + str(len(obj)) + " components"
print("Input text shape ", obj[0].shape) #max_sentences_input x max_words_input x vocab_size
print("Word graph shape ", obj[1].shape) #max_sentences_input x max_words_input x max_words_input
print("Sent graph shape ", obj[2].shape) #max_sentences_input x max_sentences_input
print("Summary shape ", obj[3].shape) #max_words_input x vocab_size