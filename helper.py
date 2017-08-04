import os
import pickle
import sys
from collections import Counter

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.manifold import TSNE

SHORT_TITLE_LENGTH = 10

MARKERS = [['red', 100, 'o'], ['black', 100, 'x'], ['cyan', 100, 'd'], ['blue', 150, '1'], ['purple', 100, 's'],
           ['green', 100, 'v'], ['yellow', 100, 'o'], ['magenta', 100, 'o'], ['orange', 100, 'o'], ['pink', 100, 'o'],
           ['brown', 100, 'o'], ['darkgreen', 100, 'o']]


def show_lib_versions():
	print('Python Interpreter version:%s' % sys.version[:3])
	print('numpy version', np.__version__)
	print('tensorflow version', tf.__version__)
	print('sklearn version', sklearn.__version__)
	print('fasttext version', fasttext.__VERSION__)


def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)


# Read lines into a list of strings.
def read_lines(filename):
	with open(filename) as f:
		content = f.readlines()
	return [x.strip() for x in content]


# Read words into a list of strings.
def read_words(filename):
	file = open(filename, 'r')
	data = tf.compat.as_str(file.read()).split()
	return data


def save_object(folder, name, obj):
	with open(folder + name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(folder, name):
	with open(folder + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def plot_with_labels(attributes, filename, titles=None, markers=None, perplexity=25, n_iter=2000):
	print('Drawing scatter plot on [%s]...' % filename)

	if attributes.shape[1] > 2:
		print('Reducing attributes...')
		tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter)
		attributes = tsne.fit_transform(attributes)

	plt.rcParams.update({'font.size': 20})
	plt.figure(figsize=(40, 40))  # in inches

	for i in range(0, len(attributes)):
		x, y = attributes[i, :]

		if markers is None:
			plot_scatter(x, y)
		else:
			plot_scatter(x, y, marker=markers[i])

		if titles is None:
			title = str(i)
		else:
			title = titles[i].strip()[:SHORT_TITLE_LENGTH]
		plt.annotate(title,
		             xy=(x, y),
		             xytext=(5, 2),
		             textcoords='offset points',
		             ha='center',
		             va='bottom')

	plt.savefig(filename)


def plot_scatter(x, y, marker=0):
	if marker >= len(MARKERS):
		marker = len(MARKERS) - 1

	plt.scatter(x, y, color=MARKERS[marker][0], s=MARKERS[marker][1] * 3 // 2, marker=MARKERS[marker][2])


def merge_dictionary(dict1, dict2):
	dict1_counter = Counter(dict1)
	dict2_counter = Counter(dict2)
	merged_dictionary = dict1_counter + dict2_counter

	return dict(merged_dictionary)


def arg_sort(values, count, descending=False):
	# use argpartition() to efficiently sort
	if descending:
		indices = np.argpartition(-values, count)
	else:
		indices = np.argpartition(values, count)

	# since indices is not sorted (because of the partial sort), sort again
	new_scores = np.zeros([count], dtype=int)
	for i in range(0, count):
		if descending:
			new_scores[i] = -values[indices[i]]
		else:
			new_scores[i] = values[indices[i]]

	new_indices = np.argsort(new_scores)

	ids = np.zeros(count, dtype=int)
	for i in range(0, count):
		ids[i] = int(indices[int(new_indices[i])])

	return ids
