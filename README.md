# Paper2Vec - Automatic clustering and search tool for CVPR2017 Papers 
Automatic clustering and search tools for CVPR2017 papers by using fastest from Facebook Research

## Overview

This project is a sample of building paper vectors and clustering for CVPR 2017 papers. Previouslly I used word2vec and this one is for [fasttext]() from [Facebook Research]().

# Steps

## 0. Scraping

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

The input corpus we built is under [data/CVPR2016](https://github.com/jiny2001/CVPR_paper_search_tool/tree/master/data/CVPR2016) and [data/CVPR2017](https://github.com/jiny2001/CVPR_paper_search_tool/tree/master/data/CVPR2017).


## 1. Count all words' occurences and unite corpus files to build a united input corpus.

Build my Paper2Vec instance. Load multiple corpus files.
And then Replaces rare words with UNK token to build a suitable size of dictionary.

```
p2v = Paper2Vec(data_dir=args.data_dir, word_dim=args.word_dim)
p2v.add_dictionary_from_file('CVPR2016/corpus.txt')
p2v.add_dictionary_from_file('CVPR2017/corpus.txt')
p2v.build_dictionary(args.max_dictionary_words)
```

## 2. Detects phrases by their appearance frequency. Then re-build a new corpus.

Count occurences of words sequence. Unite frequent sequence words with "_".
For ex "deep learning" is now one word, "deep_learning".

```
p2v.detect_phrases(args.phrase_threshold)

p2v.create_corpus_with_phrases('corpus.txt')
p2v.convert_text_with_phrases('CVPR2017/abstract.txt', 'abstract.txt')
copyfile(args.data_dir + '/CVPR2017/paper_info.txt', args.data_dir + '/paper_info.txt')

```

## 3. Train word representation with fasttext.

You can train with "skipgram" or "cbow" by fasttext. Default dimension of vector is 75.
Also you can find similar word by calling get_most_similar_words().

```
p2v.train_words_model('corpus.txt', 'fasttext_model', model=args.train_model, min_count=args.min_count)

print('[deep_learning]', p2v.get_most_similar_words('deep_learning', 12))
```

## 4. Build paper representation vectors with fasttext.

Calculate mean vector of each words' vector in abstract and title to define the vector as a paper representation vector.

```
p2v.build_paper_vectors()
```


## 5. Reduce dimensions and then apply k-means clustering.

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

# How to use

## Find similar papers

```
python find_paper_by_paper.py "Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization" --c 5

Output:
Loaded 783 papers info.

Target: [ Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization ]

5 Papers found ---
Score:4.10, [ Hyper-Laplacian Regularized Unidirectional Low-Rank Tensor Recovery for Multispectral Image Denoising ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Chang_Hyper-Laplacian_Regularized_Unidirectional_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Chang_Hyper-Laplacian_Regularized_Unidirectional_CVPR_2017_paper.pdf

Score:5.33, [ A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_A_Non-Local_Low-Rank_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_A_Non-Local_Low-Rank_CVPR_2017_paper.pdf

Score:6.36, [ Nonnegative Matrix Underapproximation for Robust Multiple Model Fitting ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Tepper_Nonnegative_Matrix_Underapproximation_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Tepper_Nonnegative_Matrix_Underapproximation_CVPR_2017_paper.pdf

Score:6.76, [ Fractal Dimension Invariant Filtering and Its CNN-Based Implementation ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Xu_Fractal_Dimension_Invariant_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Xu_Fractal_Dimension_Invariant_CVPR_2017_paper.pdf

Score:7.64, [ On the Global Geometry of Sphere-Constrained Sparse Blind Deconvolution ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_On_the_Global_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_On_the_Global_CVPR_2017_paper.pdf
```

## Find papers by keywords

```
python find_paper_by_words.py super resolution  -c 5

Loaded 783 papers info.

Keyword(s): ['super', 'resolution']

5 Papers found ---
Score:14, [ Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Dian_Hyperspectral_Image_Super-Resolution_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Dian_Hyperspectral_Image_Super-Resolution_CVPR_2017_paper.pdf

Score:12, [ Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Ledig_Photo-Realistic_Single_Image_CVPR_2017_paper.pdf

Score:9, [ Real-Time Video Super-Resolution With Spatio-Temporal Networks and Motion Compensation ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.pdf

Score:8, [ Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Lai_Deep_Laplacian_Pyramid_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Lai_Deep_Laplacian_Pyramid_CVPR_2017_paper.pdf

Score:7, [ Simultaneous Super-Resolution and Cross-Modality Synthesis of 3D Medical Images Using Weakly-Supervised Joint Convolutional Sparse Coding ]
Abstract URL:http://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Simultaneous_Super-Resolution_and_CVPR_2017_paper.html
PDF URL:http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Simultaneous_Super-Resolution_and_CVPR_2017_paper.pdf
```






