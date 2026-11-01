# Your report

1. Data
I loaded newdataset/Reddit-stock-sentiment.csv with load_dataset_tool (load_dataset_1). It contains 847 Reddit posts and comments about stocks, labeled with three sentiment classes: negative (315), neutral (423), and positive (109). The positive class is much smaller than the others, which may cause bias. For example, the overall term frequencies will mostly reflect the language of the neutral and negative posts.

2. Document-term matrix
I built a document-term matrix with build_dtm_tool (dtm_2), keeping the stopwords. It has 4,346 terms and a sparsity of 99.533%. Before running it, I predicted that the vocabulary would be smaller than in 20 Newsgroups (35,788 terms) and that the sparsity would be higher than 99.547%. The vocabulary prediction was right, but the sparsity was actually slightly lower. This is because a Reddit post has about 20 different words on average, while a 20 Newsgroups document has about 162, roughly 8 times more. The vocabulary is also about 8 times smaller, so the two effects cancel out and the sparsity stays almost the same.

3. Term frequency
I ranked the terms by frequency with term_frequency_tool (term_freq_3). All of the top 20 terms are stopwords, which is very different from my prediction that they would be stock-related terms. This is because the stopwords were kept. Compared with 20 Newsgroups, “will” (127) and “he” (132) entered the top 20 only in this dataset. “will” may appear often because stock discussions often predict the future, and “he” may refer to people who influence the market. However, this is only a guess, because the tools cannot show the original posts. Also, unlike 20 Newsgroups, no noise words from email headers such as “edu” appeared this time.

4. Distribution
I plotted the top 20 term frequencies with visualize_result_tool. The distribution is right-skewed with a long tail and is non-linear, the same shape as in 20 Newsgroups, which matches Zipf’s law.

Top 20 term frequencies (Reddit)

5. Comparison and conclusion
Both datasets have a sparsity above 99.5%, both have their top 20 terms filled with stopwords, and both show a long-tail distribution. On the other hand, the Reddit vocabulary is much smaller, and words like “will” and “he” appear, which reflect the nature of stock discussions. The smaller vocabulary may come mainly from the sample size (fewer and shorter documents) rather than a narrower topic, as described by Heaps’ law. This could be tested by sampling the same number of words from both datasets and comparing their vocabularies. In conclusion, the same tools work on different datasets, but to get meaningful results we need preprocessing first, such as removing stopwords, using TF-IDF, or computing term frequencies separately for each sentiment class.