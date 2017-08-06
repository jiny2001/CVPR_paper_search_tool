# Paper2Vec - Automatic clustering and search tool for CVPR2017 Papers 
Automatic clustering and search tools for CVPR2017 papers by using fastest from Facebook Research

## Overview

This project is a sample of building paper vectors and clustering for CVPR 2017 papers. Previouslly I used word2vec and this one is for [fasttext]() from [Facebook Research]().

## Steps

# Step 0. Scraping

Scraping paper info (title, abstract and PDF) from [CVPR open access repositry]().
Then extract words to build a corpus. In this project, code for scraping is under crawler directory in Java, (using play framework). 

```
After exstracting text from each PDF, roufgly do below processes.

Remove "-" with "/n" to cocatnate words divided on end line.
Then replace "-" with " ".
Replace all other non character codes with " "
Convert all charachters to small capital.

Remove "one character word", (http, https and ftp urls), the, an in, on, and, of to, is, for, we, with, as, that, are, by, our, this, from, be, ca, at, us, it, has, have, been, do, does, these, those, and "et al".
Replace popular plural noun to singular noun.

Remove people's name (replace one words ending in dot with dot).

```

The input corpus we built is under [data/CVPR2016](https://raw.githubusercontent.com/jiny2001/CVPR_paper_search_tool/master/data/CVPR2016) and [data/CVPR2017](https://raw.githubusercontent.com/jiny2001/CVPR_paper_search_tool/master/data/CVPR2017).


# Step 1 Count all words' occurences and unite corpus files to build a united input corpus.

Build my Paper2Vec instance. Load multiple corpus files.
And then Replaces rare words with UNK token to build a suitable size of dictionary.

```
p2v = Paper2Vec(data_dir=args.data_dir, word_dim=args.word_dim)
p2v.add_dictionary_from_file('CVPR2016/corpus.txt')
p2v.add_dictionary_from_file('CVPR2017/corpus.txt')
p2v.build_dictionary(args.max_dictionary_words)
```

# Step 2 Detects phrases by their appearance frequency. Then re-build a new corpus.

Count occurences of words sequence. Unite frequent sequence words with "_".
For ex "deep learning" is now one word, "deep_learning".

```
p2v.detect_phrases(args.phrase_threshold)

p2v.create_corpus_with_phrases('corpus.txt')
p2v.convert_text_with_phrases('CVPR2017/abstract.txt', 'abstract.txt')
copyfile(args.data_dir + '/CVPR2017/paper_info.txt', args.data_dir + '/paper_info.txt')

```

# Step 3 Train word representation with fasttext.

You can train with "skipgram" or "cbow" by fasttext. Default dimension of vector is 75.
Also you can find similar word by calling get_most_similar_words().

```
p2v.train_words_model('corpus.txt', 'fasttext_model', model=args.train_model, min_count=args.min_count)

print('[deep_learning]', p2v.get_most_similar_words('deep_learning', 12))
```

# Step 4 Build paper representation vectors with fasttext.

Calculate mean vector of each words' vector in abstract and title to define the vector as a paper representation vector.

```
p2v.build_paper_vectors()
```


#  Step 5: Reduce dimensions and then apply k-means clustering.

Reduce 75-dim of paper vector into 2-dim by using t-SNE.

<img src="https://raw.githubusercontent.com/jiny2001/CVPR_paper_search_tool/master/sample/papers.png" width="600">

Apply t-SNE again to enforce clustering.

<img src="https://raw.githubusercontent.com/jiny2001/CVPR_paper_search_tool/master/sample/papers_r.png" width="600">

After those papers are clusterized, pick frequently used words in title and abstract.

```
cluster[0] keywords: [question, over, representations, visual, vqa, scene, structure answering]

cluster[1] keywords: [feature, room, physics, optimization, semantic, transfer, layout, estimation]

cluster[2] keywords: [binary, local, lbc, convolutional_layer, cnn, linear, weights, savings, sparse, learnable]

cluster[3] keywords: [facial, wild, three_dimensional, texture, captured, morphable, new, datasets]

cluster[4] keywords: [quality, light, fields, light_field, dense, metrics, compression, reference, distorted]

cluster[5] keywords: [tracking, position, virtual, reality, commodity, vr, experience, infrastructure, infrared]

cluster[6] keywords: [tof, material, distortion, classification, depth, time, frequency, flight]
```





