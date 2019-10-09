import os 
import cPickle as pl
import numpy as np
from absl import flags
from absl import app

import sentence_graph

#Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_input_file', "dump/train.pkl", 'pickle file with processed trianing data')
flags.DEFINE_string('val_input_file', None, 'pickle file with processed validation data')
flags.DEFINE_string('test_input_file', None, 'pickle file with processed test data')
flags.DEFINE_integer('min_freq', 5, 'minimum frequency for vocabulary')
flags.DEFINE_integer('max_words_input', 100, 'maximum words per sentence in input')
flags.DEFINE_integer('max_sentences_input', 30, 'maximum number of sentences in input')
flags.DEFINE_integer('max_words_target', 40, 'maximum words per sentence in target')
flags.DEFINE_string('dump_dir', "dump/", 'folder to dump binarized pickle files')

default_vocab_symbols = {"<START>" : 0, "<END>" : 1, "<PAD>" : 2, "<UNK>" : 3}


def make_vocab(data):
	vocab_word_to_id = {}
	vocab_word_to_freq = {}

	for key, id_ in default_vocab_symbols.items():
		vocab_word_to_id[key] = id_

	next_id = 4
	for fileid, info in data.items():
		for sentence in info['text'] :
			for word in sentence:
				if word not in vocab_word_to_id:
					vocab_word_to_id[word] = next_id
					next_id += 1
					vocab_word_to_freq[word] = 1
				else:
					vocab_word_to_freq[word] += 1
		for word in info['summary']:
			if word not in vocab_word_to_id:
				vocab_word_to_id[word] = next_id
				next_id += 1
				vocab_word_to_freq[word] = 1
			else:
				vocab_word_to_freq[word] += 1
	return vocab_word_to_id, vocab_word_to_freq


def prune_vocab(vocab_word_to_freq, min_freq):
	pruned_vocab = {}

	for key, id_ in default_vocab_symbols.items():
		pruned_vocab[key] = id_

	next_id = 4
	for word, freq in vocab_word_to_freq.items():
		if freq >= min_freq and word not in default_vocab_symbols:
			pruned_vocab[word] = next_id
			next_id += 1
	return pruned_vocab



def binarize_data(data, vocab):
	binarized_data = []
	count = 0
	for id_, instance in data.items():
		input_text = instance["text"][:FLAGS.max_sentences_input]
		dep = instance["dep"][:FLAGS.max_sentences_input]

		input_token_ids = []
		word_graphs = []
		for sentence, parse in zip(input_text, dep):
			
			sentence = sentence[:FLAGS.max_words_input]
			pad_words = FLAGS.max_words_input - len(sentence)
			sentence_ids = [vocab.get(word, default_vocab_symbols["<UNK>"]) for word in sentence] \
						+ [default_vocab_symbols["<PAD>"] for i in range(pad_words)]
			input_token_ids.append(sentence_ids)

			word_graph = np.zeros((FLAGS.max_words_input, FLAGS.max_words_input))
			for edge in parse[1:]: #first edge is just an entry which indicates the subject of the sentence
				if(edge[0] <= FLAGS.max_words_input and edge[1] <= FLAGS.max_words_input):
					word_graph[edge[0] - 1][edge[1] - 1] = 1
			word_graphs.append(word_graph)

		pad_sentences = FLAGS.max_sentences_input - len(input_text)
		input_token_ids += [[default_vocab_symbols["<PAD>"] for i in range(FLAGS.max_words_input)] for j in range(pad_sentences)]
		word_graphs += [np.zeros((FLAGS.max_words_input, FLAGS.max_words_input)) for i in range(pad_sentences)]

		sent_graph = sentence_graph.make_graph(input_text, FLAGS.max_sentences_input, vocab)


		summary = instance["summary"][:FLAGS.max_words_target]
		pad_words = FLAGS.max_words_target - len(summary)
		target_token_ids = [vocab.get(word, default_vocab_symbols["<UNK>"]) for word in summary] \
							+ [default_vocab_symbols["<PAD>"] for i in range(pad_words)]
		
		binarized_data.append([np.array(input_token_ids), np.array(word_graphs), sent_graph, np.array(target_token_ids)])
	
		count += 1
		if count % 1000 == 0:
			print(str(count) + " data instances converted....")
	return binarized_data



def main(argv):
	#Load processed data and create vocabulary
	train_data = pl.load(open(FLAGS.train_input_file, 'rb'))
	vocab, vocab_word_to_freq = make_vocab(train_data)
	vocab = prune_vocab(vocab_word_to_freq, FLAGS.min_freq)
	vocab_file = FLAGS.dump_dir + "vocab.pkl"
	pl.dump(vocab, open(vocab_file, 'wb'))

	#Process training data and save
	train_binarized = binarize_data(train_data, vocab)
	train_output_pickle_file = FLAGS.dump_dir + "train_binarized.pkl"
	pl.dump(train_binarized, open(train_output_pickle_file, 'wb'))

	#Process validation data and save
	if (FLAGS.val_input_file):
		val_data = pl.load(open(FLAGS.val_input_file, 'rb'))
		val_binarized = binarize_data(val_data, vocab)
		val_output_pickle_file = FLAGS.dump_dir + "val_binarized.pkl"
		pl.dump(val_binarized, open(val_output_pickle_file, 'wb'))

	#Process test data and save
	if (FLAGS.test_input_file):
		test_data = pl.load(open(FLAGS.test_input_file, 'rb'))
		test_binarized = binarize_data(test_data, vocab)
		test_output_pickle_file = FLAGS.dump_dir + "test_binarized.pkl"
		pl.dump(test_binarized, open(test_output_pickle_file, 'wb'))


if __name__ == '__main__':
   app.run(main)




