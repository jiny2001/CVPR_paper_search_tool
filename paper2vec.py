import collections

import fasttext
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import helper

UNK = 'UNK'  # unknown words (rare words)
EOS = '<eos>'  # end of sentence
EOP = '<eop>'  # end of paper
LABEL_PREFIX = '__label__'


class PaperInfo:
	def __init__(self, title, abstract_url, pdf_url):
		self.title = title
		self.abstract_url = abstract_url
		self.pdf_url = pdf_url

		self.abstract_freq = dict()


class Paper2Vec:
	def __init__(self, word_dim=100, data_dir='data'):

		self.word_dim = word_dim
		self.data_dir = data_dir + '/'

		self.words = []
		self.papers = 0
		self.dictionary_words = 0
		self.index_UNK = -1
		self.index_EOS = -1
		self.index_EOP = -1

		helper.make_dir(self.data_dir)

	def add_dictionary_from_file(self, filename):

		filename = self.data_dir + filename
		words = helper.read_words(filename)
		print('File:[%s] Words:%s ' % (filename, '{:,}'.format(len(words))))
		self.words += words

	def build_dictionary(self, max_dictionary_words):

		self.count = [(UNK, -1)]
		self.count.extend(collections.Counter(self.words).most_common(max_dictionary_words - 1))
		self.dictionary = dict()
		for word, _ in self.count:
			index_word = len(self.dictionary)
			self.dictionary[word] = index_word
			if word == EOS:
				self.index_EOS = index_word
			if word == EOP:
				self.index_EOP = index_word

		self.data = list()  # indexed words data
		unk_count = 0
		for word in self.words:
			if word in self.dictionary:
				index = self.dictionary[word]
			else:
				index = 0  # dictionary['UNK']
				unk_count += 1
			self.data.append(index)

		self.index_UNK = 0
		self.count[self.index_UNK] = (UNK, unk_count)
		self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

		self.dictionary_words = len(self.dictionary)

		print('Finished building dictionary. Size:%s %s words are assigned as unk.' %
		      ('{:,}'.format(len(self.dictionary)), '{:,}'.format(unk_count)))

	def detect_phrases(self, phrase_threshold):

		self.phrase_counter = np.zeros([self.dictionary_words, self.dictionary_words], dtype='int32')
		self.phrase_threshold = phrase_threshold

		pre_index = 0
		for i in range(0, len(self.data)):
			index = self.data[i]

			# since index_UNK is UNK and index_EOS is <eos>, we ignore those words.
			if index != self.index_UNK and index != self.index_EOS and pre_index != self.index_UNK and pre_index != self.index_EOS:
				self.phrase_counter[pre_index, index] += 1
			pre_index = index

		for y in range(1, self.dictionary_words):
			if self.count[y][1] < self.phrase_threshold:
				break
			for x in range(1, self.dictionary_words):
				if self.count[x][1] < self.phrase_threshold:
					break
				if self.phrase_counter[x, y] > self.phrase_threshold:
					print(self.reverse_dictionary[x], self.reverse_dictionary[y], ':', self.phrase_counter[x, y])

	def create_corpus_with_phrases(self, filename):

		filename = self.data_dir + filename
		print('Building new corpus on [%s]...' % filename)
		target = open(filename, 'w+')
		words = self.words
		pre_index = -1

		for word in words:
			self.write_with_phrases(word, pre_index, target)
			if word in self.dictionary:
				pre_index = self.dictionary[word]
			else:
				pre_index = self.index_UNK

		target.close()

	def write_with_phrases(self, word, pre_index, target):

		if pre_index != -1:
			if word in self.dictionary:
				index = self.dictionary[word]
				if index == self.index_EOS:
					pass
				elif self.phrase_counter[pre_index, index] > self.phrase_threshold:
					target.write('_')
				else:
					target.write(' ')
			else:
				target.write(' ')

		if word in self.dictionary:
			index = self.dictionary[word]
			if index == self.index_EOS:
				target.write(' ')
			else:
				target.write(word)
		else:
			target.write(UNK)

	def convert_text_with_phrases(self, src_filename, dest_filename):

		src_filename = self.data_dir + src_filename
		dest_filename = self.data_dir + dest_filename
		print('Building new text file on [%s]...' % src_filename)

		words = helper.read_words(src_filename)
		target = open(dest_filename, 'w+')
		pre_index = 0

		target.write(words[0])
		for word in words[1:]:
			if word in self.dictionary:
				index = self.dictionary[word]
				if self.phrase_counter[pre_index, index] > self.phrase_threshold:
					target.write('_')
				else:
					target.write(' ')
			else:
				index = 0  # dictionary['UNK']
				target.write(' ')

			target.write(word)
			pre_index = index

		target.close()

	def create_label(self, src_filename, dest_filename):

		src_filename = self.data_dir + src_filename
		dest_filename = self.data_dir + dest_filename
		print('Building new label file on [%s]...' % dest_filename)

		words = helper.read_words(src_filename)
		target = open(dest_filename, 'w+')

		paper = 0
		pre_word_is_eol = True

		for word in words:

			if word == EOP:
				paper += 1
			elif word == EOS:
				target.write('.')
				pre_word_is_eol = True
				target.write('\n')
			else:
				if pre_word_is_eol:
					target.write(LABEL_PREFIX + str(paper))
					pre_word_is_eol = False
				target.write(' ' + word)

		target.close()

	def train_words_model(self, corpus_filename, model_filename, model='skipgram', min_count=5):

		corpus_filename = self.data_dir + corpus_filename
		model_filename = self.data_dir + model_filename
		print('Training for [%s] Model=%s Dim=%d MinCount=%d...' % (corpus_filename, model, self.word_dim, min_count))

		if model == 'skipgram':
			self.model = fasttext.skipgram(input_file=corpus_filename, output=model_filename, dim=self.word_dim,
			                               min_count=min_count)
		elif model == 'cbow':
			self.model = fasttext.cbow(input_file=corpus_filename, output=model_filename, dim=self.word_dim,
			                           min_count=min_count)
		else:
			print('model param should be cbow or skipgram')
			return

		self.words_list = list(self.model.words)

		print('Finished. Dictionary size:%s' % '{:,}'.format(len(self.model.words)))

	def load_words_model(self, model_filename):

		model_filename = self.data_dir + model_filename
		self.model = fasttext.load_model(model_filename + '.bin')
		self.word_dim = self.model.dim
		self.words_list = list(self.model.words)
		print('Loaded. Dictionary size:%s' % '{:,}'.format(len(self.model.words)))

	def get_most_similar_words(self, target_word, count):

		# build word-vector array and word-score array
		target_vector = np.array(self.model[target_word])
		scores = np.zeros(len(self.model.words))

		for i in range(len(self.words_list)):
			scores[i] = np.mean(np.square(np.array(self.model[self.words_list[i]]) - target_vector))

		# use argpartition() to efficiently sort
		indices = np.argpartition(scores, count + 1)[:count + 1]

		# since indices is not sorted (because of the partial sort), sort again
		new_scores = np.zeros([count + 1])
		for i in range(0, count + 1):
			new_scores[i] = scores[indices[i]]

		new_indices = np.argsort(new_scores)

		str = ''
		for i in range(1, len(new_indices)):
			word_index = int(indices[int(new_indices[i])])
			str += '%s:%2.3f, ' % (self.words_list[word_index], scores[word_index])

		return str

	def load_paper_info(self):

		paper_info_lines = helper.read_lines(self.data_dir + '/paper_info.txt')

		self.papers = len(paper_info_lines) // 3  # each paper contains 3 lines (info, abstract_url, paper_url)
		self.paper = self.papers * [None]  # type: list[PaperInfo]

		for i in range(self.papers):
			self.paper[i] = PaperInfo(paper_info_lines[i * 3], paper_info_lines[i * 3 + 1], paper_info_lines[i * 3 + 2])

	def build_paper_vectors(self):

		self.load_paper_info()

		# load paper abstract and build paper representation vector
		abstract_words = helper.read_words(self.data_dir + '/abstract.txt')

		self.paper_vectors = np.zeros([self.papers, self.word_dim])
		papers = 0
		vectors = 0

		for word in abstract_words:
			if word == EOS:
				pass
			elif word == EOP:
				self.paper_vectors[papers] /= vectors
				papers += 1
				vectors = 0
			else:
				self.paper_vectors[papers] += self.model[word]
				vectors += 1
				if word in self.paper[papers].abstract_freq:
					self.paper[papers].abstract_freq[word] = self.paper[papers].abstract_freq[word] + 1
				else:
					self.paper[papers].abstract_freq[word] = 1

	def reduce_paper_vectors_dim(self, new_dim, perplexity=25):

		print('Reducing paper vectors from %d to %d dim...' % (self.paper_vectors.shape[1], new_dim))
		tsne = TSNE(perplexity=perplexity, n_components=new_dim, init='pca', n_iter=5000)
		self.paper_vectors = tsne.fit_transform(self.paper_vectors)

	def clustering_papers(self, clusters=10):

		estimator = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
		estimator.fit(self.paper_vectors)
		self.paper_cluster_ids = estimator.labels_

		self.cluster_abstract_freq = []

		for i in range(clusters):
			abstract = dict()
			for j in range(self.papers):
				if self.paper_cluster_ids[j] == i:
					abstract = helper.merge_dictionary(abstract, self.paper[i].abstract_freq)

			self.cluster_abstract_freq.append(sorted(abstract.items(), key=lambda x: x[1], reverse=True))

	def save_paper_vectors(self):

		helper.save_object(self.data_dir, 'paper_vectors', self.paper_vectors)
		helper.save_object(self.data_dir, 'paper_info', self.paper)

		print('Saved %d papers info.' % self.papers)

	def load_paper_vectors(self):

		self.paper_vectors = helper.load_object(self.data_dir, 'paper_vectors')
		self.paper = helper.load_object(self.data_dir, 'paper_info')
		self.papers = min(self.paper_vectors.shape[0], len(self.paper))

		print('Loaded %d papers info.' % self.papers)

	def find_similar_papers(self, paper_id, count=5):

		target_vector = self.paper_vectors[paper_id]

		scores = np.zeros(self.papers)

		for i in range(self.papers):
			scores[i] = np.mean(np.square(np.array(self.paper_vectors[i]) - target_vector))

		ids = helper.arg_sort(scores, count + 1)[1:]
		results = []

		for i in range(len(ids)):
			if scores[ids[i]] <= 0:
				break
			else:
				results.append((ids[i], scores[ids[i]]))

		return results

	def find_by_paper_title(self, title):

		title = title.lower()

		for i in range(self.papers):
			if title in self.paper[i].title.lower():
				return i

		return -1

	def find_by_keywords(self, keywords, count):

		scores = np.zeros(self.papers)
		for i in range(self.papers):

			for keyword in keywords:
				keyword = keyword.lower()
				if keyword in self.paper[i].abstract_freq:
					scores[i] += self.paper[i].abstract_freq[keyword]
				else:
					scores[i] = 0
					break

		ids = helper.arg_sort(scores, count, descending=True)
		results = []

		for i in range(len(ids)):
			if scores[ids[i]] <= 0:
				break
			else:
				results.append((ids[i], scores[ids[i]]))

		return results
