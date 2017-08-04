import argparse

from paper2vec import Paper2Vec

parser = argparse.ArgumentParser()

parser.add_argument('title', help='paper\'s title')
parser.add_argument('-c', '--count', default=5, type=int, nargs='?', help='num of papers to find')
parser.add_argument('--data_dir', default='data', type=str, nargs='?', help='directory for data')
args = parser.parse_args()


def main(args):

	p2v = Paper2Vec(data_dir=args.data_dir)
	p2v.load_paper_vectors()

	paper_id = p2v.find_by_paper_title(args.title)
	if paper_id < 0:
		print('Can\'t find paper from [%s].' % args.title)
		exit()

	print('\nTarget: [ %s ]' % p2v.paper[paper_id].title)
	results = p2v.find_similar_papers(paper_id, args.count)

	print('\n%d Papers found ---' % len(results))

	for result in results:
		# result[0] contains paper id, result[1] contains matching score (smaller is better)
		print('Score:%0.2f, [ %s ]' % (result[1], p2v.paper[result[0]].title))
		print('Abstract URL:%s' % p2v.paper[result[0]].abstract_url)
		print('PDF URL:%s\n' % p2v.paper[result[0]].pdf_url)


if __name__ == '__main__':
	main(args)
