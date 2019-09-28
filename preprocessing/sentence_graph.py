import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_sentence_vector(sentence, vocab, dim):
	vec = [0 for i in range(dim)]
	for word in sentence:
		vec[vocab[word]] = 1
	return vec


def make_graph(sentences, num_nodes, vocab, min_sim = 0.25):
	#Using BOW-cosine similarity
	num_sentences = len(sentences)
	dim = len(vocab)
	sent_graph = np.zeros((num_nodes, num_nodes))

	for i, sent1 in enumerate(sentences):
		for j, sent2 in enumerate(sentences):
			if i == j:
				continue
			vec1 = get_sentence_vector(sent1, vocab, dim)
			vec2 = get_sentence_vector(sent2, vocab, dim)
			cosine_sim = cosine_similarity([vec1, vec2])[0][0]
			if cosine_sim >= min_sim:
				sent_graph[i][j] = cosine_sim

	return sent_graph
