from stanfordcorenlp import StanfordCoreNLP
import os, logging
import cPickle as pl
from absl import flags
from absl import app

#Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', "fake_data/", 'folder with data files to parse')
flags.DEFINE_string('dump_dir', "fake_dump/", 'folder to dump dependency parses into')
flags.DEFINE_string('name', 'data.pkl', 'output pickle file name')
flags.DEFINE_string('corenlp', 'corenlp', 'path to the stanford corenlp folder')


def extract_data(textlines):
	#print(textlines)
	introduction_indicator = "[XSUM]INTRODUCTION[XSUM]"
	body_indicator = "[XSUM]RESTBODY[XSUM]"
	unwanted_lines = ["Share this with",
					  "Email",
					  "Facebook",
					  "Messenger",
					  "Twitter",
					  "Pinterest",
					  "WhatsApp",
					  "LinkedIn",
					  "Copy this link",
					  "These are external links and will open in a new window"
					  ]

	line_count = 0
	for line in textlines:
		if line.strip() == introduction_indicator:
			break
		line_count += 1
	summary = textlines[line_count + 1].strip()
	remaning_textlines = textlines[line_count + 2:]
	for i, line in enumerate(remaning_textlines):
		if line.strip() == body_indicator or line.strip() in unwanted_lines:
			body_start = i + 1
	text = [line.strip() for line in remaning_textlines[body_start:]]

	return text, summary



def main(argv):
	#Option 1
	#Before running this file, run the following command from terminal after downloading the 'corenlp' folder: 
	#java -mx4g -cp "corenlp/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 8000 -timeout 15000 -annotators "tokenize,ssplit,dep"
	nlp = StanfordCoreNLP('http://localhost', port=8000, logging_level=logging.WARNING)
	
	#Option 2
	#nlp = StanfordCoreNLP(FLAGS.corenlp)

	print "Test text\n"
	text = 'Guangdong University of Foreign Studies is located in Guangzhou.'
	print nlp.dependency_parse(text)

	#Produce parses for all files in given folder
	text_files =  os.listdir(FLAGS.data_dir)
	num_files = len(text_files)
	print("Beginning to parse " + str(num_files) + " files.")
	count = 0
	max_words_input = -1
	max_words_target = -1
	max_sentences_input = -1
	data = {}

	for text_file in text_files:
		text, summary = extract_data(open(FLAGS.data_dir + text_file).readlines())
		
		dep_parse = []
		tokenized_text = []
		for sentence in text:
			sentence_parse = []
			tokens = nlp.word_tokenize(sentence)
			for triplet in nlp.dependency_parse(sentence):
				#triplet is (relation, goernor_index, dependent_index)
				#ignore direction and relation identifier of dependency
				sentence_parse.append((triplet[1], triplet[2])) 
			dep_parse.append(sentence_parse)
			tokenized_text.append(tokens)

			max_words_input = max(len(tokens), max_words_input)
		
		fileid = text_file[:text_file.find(".")]
		tokenized_summary = nlp.word_tokenize(summary)
		data[fileid] = {"text": tokenized_text, "summary": tokenized_summary, "dep": dep_parse}

		max_sentences_input = max(len(text), max_sentences_input)
		max_words_target = max(len(tokenized_summary), max_words_target)
		
		count += 1
		if count % 500 == 0:
			print(str(count) + " files parsed....")

	print("Done, " + str(count) + " files parsed, saving....")
	print("Maximum number of words per sentence in input files: " + str(max_words_input))
	print("Maximum number of words in target: " + str(max_words_target))
	print("Maximum number of sentences per input file: " + str(max_sentences_input))
	pl.dump(data, open(FLAGS.dump_dir + FLAGS.name, 'wb'))
	print("Saved")

	nlp.close() # Do not forget to close! The backend server will consume a lot memory.
   

if __name__ == '__main__':
   app.run(main)




