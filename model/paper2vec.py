
import numpy as np
from model import helper

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
	def __init__(self, model_dir='data'):
		self.model_dir = model_dir + '/'
		self.paper_vectors = None
		self.paper = None
		self.papers = 0
		self.paper_cluster_ids = None

	def save_paper_vectors(self):

		helper.save_object(self.model_dir, 'paper_vectors', self.paper_vectors)
		helper.save_object(self.model_dir, 'paper_info', self.paper)
		helper.save_object(self.model_dir, 'cluster_ids', self.paper_cluster_ids)

		print('Saved %d papers info.' % self.papers)

	def load_paper_vectors(self, load_cluster_ids = False):

		self.paper_vectors = helper.load_object(self.model_dir, 'paper_vectors')
		self.paper = helper.load_object(self.model_dir, 'paper_info')
		if load_cluster_ids:
			self.paper_cluster_ids = helper.load_object(self.model_dir, 'cluster_ids')
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

	def find_by_keywords(self, keywords, count = 0):

		if count <= 0:
			count = self.papers - 1

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
