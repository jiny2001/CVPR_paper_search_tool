import os
import pickle
from collections import Counter

import numpy as np

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
	with open(filename) as f:
		words = f.read().split()
	return words


def save_object(folder, name, obj):
	with open(folder + name + '.pkl', 'wb+') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(folder, name):
	with open(folder + name + '.pkl', 'rb') as f:
		return pickle.load(f)


def merge_dictionary(dict1, dict2):
	dict1_counter = Counter(dict1)
	dict2_counter = Counter(dict2)
	merged_dictionary = dict1_counter + dict2_counter

	return dict(merged_dictionary)


def arg_sort(values, count=0, descending=False):
	if count <= 0:
		count = len(values)

	# #	use argpartition() to efficiently sort (if np is latest enough)
	# if descending:
	# 	indices = np.argpartition(-values, count)
	# else:
	# 	indices = np.argpartition(values, count)

	# # since indices is not sorted (because of the partial sort), sort again
	# new_scores = np.zeros([count], dtype=int)
	# for i in range(0, count):
	# 	if descending:
	# 		new_scores[i] = -values[indices[i]]
	# 	else:
	# 		new_scores[i] = values[indices[i]]
	#
	# new_indices = np.argsort(new_scores)
	#
	# ids = np.zeros(count, dtype=int)
	# for i in range(0, count):
	# 	ids[i] = int(indices[int(new_indices[i])])

	if descending:
		indices = np.argsort(-values)
	else:
		indices = np.argsort(values)

	ids = np.zeros(count, dtype=int)
	for i in range(0, count):
		ids[i] = int(indices[i])

	return ids
