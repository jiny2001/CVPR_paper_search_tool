import pickle
import collections
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from sklearn.manifold import TSNE

#paper finder tool
print "[paper finder tools]"
print "numpy version", np.__version__
print "tensorflow version", tf.__version__

use_phrase = True
vocabulary_size = 8000

if use_phrase:
  title_filename = 'title_words_2.txt'
  sample_word = "deep_learning"
  obj_folder = "obj"
  abstract_filename = "abstract_2.txt"
else:
  title_filename = 'title_words.txt'
  sample_word = "deep"
  obj_folder = "obj_no_phrase"
  abstract_filename = "abstract.txt"

import helper

# step1 load data set and build initial paper vector

embeddings = helper.load_object(obj_folder, "final_embeddings")
dictionary = helper.load_object(obj_folder, "dictionary")
reverse_dictionary = helper.load_object(obj_folder, "reverse_dictionary")

print "Dictionary Size:%d" % len(dictionary)
papers = helper.Papers(embeddings)

papers.build_paper_vectors(title_filename, abstract_filename, dictionary, embeddings)


#step2 test count words
sample_list = ["deep", "deep_neural_network", "learning", "deep_learning", "qlearning", "cnn", "convolutional", "convnet", "conv", "dcnn", "rnn", "recurrent", "lstm", "long_short_term_memory", "dnn", "dbn", "autoencoder", "auto_encoder", "neural", "neural_network", "fully_connected", "attention_network"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["cnn", "convolutional", "convnet", "conv", "dcnn"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["markov", "mrf", "mrfs", "crf", "crfs", "random_field", "random_fields"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["optical_flow", "optical_flows"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["image_net", "imagenet"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["lfw"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["sfm","structure_motion","structure_from","structure_from_motion","motion_structure", "mfs"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["mnist"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["three_dimensional", "three_dimensional_point", "pointcloud", "point_cloud", "depth", "rgbd", "stereo"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["hog", "sift", "surf", "feature_extraction", "feature", "feature_detection", "saliency", "salient", "salience", "saliences"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["svm", "support_vector", "logistic_regression"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["pose_estimation"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["light_field"]
print sample_list[0],papers.count_words_containing_papers(sample_list)

sample_list = ["unsupervised"]
print sample_list[0],papers.count_words_containing_papers(sample_list)


#step3 set attribute
sample_list = ["deep", "deep_neural_network", "learning", "deep_learning", "qlearning", "cnn", "convolutional", "convnet", "conv", "dcnn", "rnn", "recurrent", "lstm", "long_short_term_memory", "dnn", "dbn", "autoencoder", "auto_encoder", "neural", "neural_network", "fully_connected", "attention_network"]
papers.set_attribute(sample_list,1)

sample_list = ["three_dimensional", "three_dimensional_point", "pointcloud", "point_cloud", "depth", "rgbd", "stereo"]
papers.set_attribute(sample_list,2)

sample_list = ["markov", "mrf", "mrfs", "crf", "crfs", "random_field", "random_fields"]
papers.set_attribute(sample_list,3)

sample_list = ["optical_flow", "optical_flows"]
papers.set_attribute(sample_list,4)

sample_list = ["hog", "sift", "surf", "feature_extraction", "feature", "feature_detection", "saliency", "salient", "salience", "saliences"]
papers.set_attribute(sample_list,5)

sample_list = ["svm", "support_vector", "logistic regression"]
papers.set_attribute(sample_list,6)

papers.determine_attribute()

for i in range(0, 60):
  print papers.paper[i].short_title, papers.paper[i].attribute, papers.paper[i].attribute_vector

#papers.build_graph()
#papers.build_graph2()
#papers.build_graph3()
papers.build_graph_ours()
papers.build_graph()

papers.find_by_paper(1-1)
papers.find_by_paper(575-1)
papers.find_by_words(["three_dimensional","unsupervised"])
