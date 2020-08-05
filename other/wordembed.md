

#### 测试语料details：

| 测试语料   | 样本个数 | 评估指标 |
| ---------- | -------- | -------- |
| toefl      | 80       | Accuracy |
| men        | 1000     | Spearman |
| rg         | 65       | Spearman |
| word353    | 353      | Spearman |
| word353sim | 2        | Spearman |
| word353rel | 252      | Spearman |



#### open 的结果：

<img src="/Users/aszzy/Documents/study/note/other/vec.png" height="200px">

#### 测试结果

|              | rg        | ws     | wss        | wsr    | men    | toefl     |
| ------------ | --------- | ------ | ---------- | ------ | ------ | --------- |
| cbow         | 0.7398    | 0.6292 | 0.7167     | 0.5457 | 0.7072 | 0.692     |
| glove_50     | 0.5570    | 0.5027 | 0.5418     | 0.5    | 0.6731 | 0.718     |
| tscca        | 0.5578    | 0.5416 | 0.6599     | 0.4314 | 0.5427 | 0.6       |
| hpca         | 0.1979    | 0.2782 | 0.3797     | 0.1422 | 0.2013 | 0.557     |
| random_proj  | 0.1705    | 0.1870 | 0.2044     | 0.1615 | 0.1231 | 0.532     |
| ==MAX_10==   | 0.7522    | 0.6028 | 0.7483     | 0.4945 | 0.6399 | **0.803** |
| 5G==Max_10== | **0.785** | 0.6123 | **0.7553** | 0.5155 | 0.6524 | 0.74      |
| pw_4         | 0.7344    | 0.6628 | 0.7603     | 0.6057 | 0.678  | 0.714     |
|              |           |        |            |        |        |           |





adapter sampler

|       | ws   | wss  | wsr  | rg   | rw   | simlex | mturk |
| ----- | ---- | ---- | ---- | ---- | ---- | ------ | ----- |
| sga   | 0.7  | 0.65 | 0.32 | 0.71 | 0.36 | 0.47   | 0.73  |
| cbowa | 0.69 | 0.66 | 0.33 | 0.72 | 0.38 | 0.49   | 0.74  |
|       |      |      |      |      |      |        |       |

|       | semantic | syntactic | total |
| ----- | -------- | --------- | ----- |
| cobwa | 0.793    | 0.721     | 0.779 |
| sga   | 0.868    | 0.798     | 0.823 |



wordRank

|          | ws   | wss   | wsr   | men   | rw    | simlex | mturk |
| -------- | ---- | ----- | ----- | ----- | ----- | ------ | ----- |
| word2vec |      | 0.739 | 0.609 | 0.754 | 0.455 | 0.366  | 0.664 |
| glove    |      | 0.757 | 0.675 | 0.788 | 0.436 | 0.416  | 0.697 |
| wordrank |      | 0.794 | 0.705 | 0.781 | 0.474 | 0.435  | 0.635 |

|          | semantic | syntactic |
| -------- | -------- | --------- |
| word2vec | 0.788    | 0.720     |
| glove    | 0.809    | 0.711     |
| wordrank | 78.4     | 0.747     |





**2.2.1 将采样(subsampling)**

降采样越低，对高频词越不利，对低频词有利。可以这么理解，本来高频词 词被迭代50次，低频词迭代10次，如果采样频率降低一半，高频词失去了25次迭代，而低频词只失去了5次。一般设置成le-5。个人觉得，降采样有类似tf-idf的功能，降低高频词对上下文影响的权重。

**2.2. 2 语言模型**

skip-gram 和cbow,之前有对比，切词效果偏重各不相同。
从效果来看，感觉cbow对词频低的词更有利。这是因为 cbow是基于周围词来预测某个词，虽然这个词词频低，但是他是基于 周围词训练的基础上，通过算法来得到这个词的向量。通过周围词的影响，周围词训练的充分，这个词就会收益。

**2.2. 3 窗口大小**

窗口大小影响 词 和前后多少个词的关系，和语料中语句长度有关，建议可以统计一下语料中，句子长度的分布，再来设置window大小。一般设置成8。

**2.2. 4 min-count**

最小词频训练阀值，这个根据训练语料大小设置，只有词频超过这个阀值的词才能被训练。

根据经验，如果切词效果不好，会切错一些词，比如 “在深圳”，毕竟切错的是少数情况，使得这种错词词频不高，可以通过设置相对大一点的 min-count 过滤掉切错的词。

**2.2. 5 向量维度**

如果词量大，训练得到的词向量还要做语义层面的叠加，比如 句子 的向量表示 用 词的向量叠加，为了有区分度，语义空间应该要设置大一些，所以维度要偏大。一般 情况下200维够用。







**阶段性的结论，说得通，为什么不好**





正样本+权重

#### Negative Sampling

SGNS is implicitly performing the weighted factorization of a shifted PMI matrix (Levy and Goldberg, 2014). Window sampling ensures the factorization weights frequent co-occurrences heavily, but also takes into account negative co-occurrences, thanks to negative sampling.

Cbow/Skip-Gram 是一个local context window的方法，比如使用NS来训练，缺乏了整体的词和词的关系，负样本采用sample的方式会缺失词的关系信息。 
另外，直接训练Skip-Gram类型的算法，很容易使得高曝光词汇得到过多的权重

#### GLOVE

The weighting function (in eq. (3)) penalizes more heavily reconstruction error of frequent co-occurrences, improving on PPMI-SVD’s L2 loss, which weights all reconstruction errors equally. However, as it does not penalize reconstruction errors for pairs with zero counts in the co-occurrence matrix, no effort is made to scatter the vectors for these pairs.

Global Vector融合了矩阵分解Latent Semantic Analysis (LSA)的全局统计信息和local context window优势。融入全局的先验统计信息，可以加快模型的训练速度，又可以控制词的相对权重。

#### LexVec

加权策略：与focal-loss思路相反





分布式词向量：混淆语法相关和语义相关

因此，分布向量之间的相似性仅表示抽象语义关联，而不是精确的语义关系，For example, it is difficult to discern
synonyms from antonyms in distributional spaces.

Specialization models：external lexical knowledge(wordNet, BabelNet)

* 在目标函数中加入约束
* finetune通用词向量，满足约束









语法相关性跟语义相关性跟窗口的关系，2，20

测试集是语义相关多，还是语法相关多

用word2vec预训练，在fine-tune

尝试了什么，为什么要尝试，结果不好，而不是瞎试，结果不好

看5-10论文，看motivation，而不是method，他们为什么要这么做，解决的什么问题



加上位置向量

负样本保留交叉熵，正样本的监督信息改成ppMi矩阵，换成max-margin loss，









word2vec：

* 采样分布可能偏离真实分布，分布在训练过程中是变化的，全局的采样分布不符合实际情况
* 没有利用统计信息，01监督信息太绝对
* 没有考虑证样本的权重，位置信息
* 无法确保证样本的score一定大于负样本

Glove：

* 没有考虑负样本

#### Matrix Factorization using Window Sampling and Negative Sampling for Improved Word Representations：

![image-20181026101638191](/Users/aszzy/Library/Application Support/typora-user-images/image-20181026101638191.png)

* 在semantic similarity tasks上PPMI矩阵由于PMI矩阵，Lexvec显式分解PPMI矩阵。给证样本较高的权重，同时考虑负样本



#### Enhancing the LexVec Distributed Word Representation Model Using Positional Contexts and External Memory

* 给LexVec加了位置信息
* 预处理采样文件



#### Improving Negative Sampling for Word Representation using Self-embedded Features






#### Spectral Word Embedding with Negative Sampling

* Bag-of-Word: 没有利用预料中的语义和句法关系，而且丢失了词序信息
* 



