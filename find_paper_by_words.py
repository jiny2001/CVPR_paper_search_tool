import argparse

from paper2vec import Paper2Vec

parser = argparse.ArgumentParser()

parser.add_argument('keywords', nargs='+', help='keywords for search')
parser.add_argument('-c', '--count', default=5, type=int, nargs='?', help='max num of papers to find')
parser.add_argument('--data_dir', default='data', type=str, nargs='?', help='directory for data')
args = parser.parse_args()


def main(args):
	p2v = Paper2Vec(data_dir=args.data_dir)
	p2v.load_paper_vectors()

	print('\nKeyword(s):', args.keywords)
	results = p2v.find_by_keywords(args.keywords, args.count)

	if len(results) <= 0:
		print('No papers found.')
		exit(0)

	print('\n%d Papers found ---' % len(results))

	for result in results:
		# result[0] contains paper id, result[1] contains matching score (larger is better)
		print('Score:%d, [ %s ]' % (result[1], p2v.paper[result[0]].title))
		print('Abstract URL:%s' % p2v.paper[result[0]].abstract_url)
		print('PDF URL:%s\n' % p2v.paper[result[0]].pdf_url)


if __name__ == '__main__':
	main(args)
