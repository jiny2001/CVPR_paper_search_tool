import argparse

import helper
from paper2vec import Paper2Vec

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', default='data', type=str, nargs='?', help='directory for data')

parser.add_argument('-x', '--perplexity', default=25, type=int, nargs='?', help='perplexity param for t-SNE')
parser.add_argument('-c', '--clusters', default=9, type=int, nargs='?', help='number of clusters to be divided')

args = parser.parse_args()


def main(args):

	p2v = Paper2Vec(data_dir=args.data_dir)
	p2v.load_words_model('fasttext_model')

	print('\nStep 5: Build paper representation with fasttext.')
	p2v.build_paper_vectors()

	print('\nStep 6: Reduce dimensions and then apply k-means clustering.')
	p2v.reduce_paper_vectors_dim(2, perplexity=args.perplexity)
	p2v.clustering_papers(clusters=args.clusters)
	helper.plot_with_labels(p2v.paper_vectors, filename=args.data_dir+'/papers.png', markers=p2v.paper_cluster_ids,
	                        perplexity=args.perplexity)

	# for visualizing, do dim reducing twice
	p2v.reduce_paper_vectors_dim(2, perplexity=args.perplexity)
	p2v.clustering_papers(clusters=args.clusters)
	helper.plot_with_labels(p2v.paper_vectors, filename=args.data_dir+'/papers_r.png', markers=p2v.paper_cluster_ids,
	                        perplexity=args.perplexity)


if __name__ == '__main__':
	main(args)
