multitask

* 利用外部数据，semantic gap，多任务组合

feature-based

* 特征选择
* 词对特征，在不同pattern下有不同关系

神经网络

* 通用模型，没发结合每种关系的特点
* 



<font color='red'>两句话一起决定了篇章的主题，根据这个主题才能判断哪些词重要，哪些不重要（找个例子）；同时应该考虑短语对的关系，而不是词对的关系，因为加上一个修饰词（如否定），可能会产生不同的语义关系（再找个例子）。即：1）确定主题，2）选择重要的短语。再加上惩罚项，比如稀疏约束</font>







we use sections 2-20 for training, sections 21-22 for testing and sections 0-1 for development set.

Sections 2–20, Sections 0-1 & Sections 23-24, and Sections 21-22 for training, development and testing, respectively. we use Stanford
CoreNLP [Manning et al., 2014] for tokenization, pad all sentences to length 50, and use Stanford’s GloVe [Pennington et al., 2014] 100 dimensional pre-trained word embeddings for SWIM and 50 dimensional pre-trained embedding for BiLSTMs + SWIM. The embedding layer is fixed during training, and dropout is performed on the input and MLP layer (dropout percentage = 20%). For training, we adopt multi-class cross-entropy loss, using AdaGrad for the stochastic optimization [Duchi et al., 2011]. The initial learning rate is set at 0.01, with a batch size of 32. Following [Liu and Li, 2016; Rutherford and Xue, 2014], we use instance re-weighting.





![image-20181019201140410](/Users/aszzy/Library/Application Support/typora-user-images/image-20181019201140410.png)

尝试把互信息作为特征加到网络里

使用sharp的tool抽取语料













因此，计算话语连词的语境差异涉及从该话语连词所约束的所有隐含话语关系中拟合单词分布，并使同一话语连词所约束的所有显性话语关系中的另一个与之相符。



comparison：

反义词，Negation for Comparison is usually used to express the contradiction towards the same topic

contingency：

1）主观性：由人主观的给出结论，然后加以论证。

2）意图：agent给出意图，然后给出原因





##### 实验结果

##### 1 Baseline

##### 两个单独的CNN，池化，分类，没有att，没有mlp，

*note：括号内为min_count*

| comparison | contingency | expansion | expan+ | temporal |
| ---------- | ----------- | --------- | ------ | -------- |
| 0.3830     | 0.5666      | 0.7110    | 0.7972 | 0.3008   |



For the second setting, to solve the problem of unbalanced classes in the training data, we follow the reweighting method of (Rutherford and Xue, 2015) to reweigh the training instances according to the size of each relation class

Improving the inference of implicit discourse relations via classifying explicit discourse connectives

macro-averaged F1 and Accuracy



|           | comparison | contingency | expansion | expan+ | temporal     |
| --------- | ---------- | ----------- | --------- | ------ | ------------ |
| LSTM      | 0.3293     | 0.5308      | 0.6721    | 0.8002 | 0.26-0.3 (1) |
| Bi-LSTM   | 0.3360     | 0.5416      | 0.6748    | 0.7965 | 0.26-0.3 (1) |
| Attention | 0.3668     | 0.5609      | 0.6954    | 0.7972 | 0.2961       |





##### Att-LSTM+单句分类：

| Relation    | F1-score (15次实验取均值)                                    | Average |
| ----------- | ------------------------------------------------------------ | ------: |
| comparison  | 0.3525, 0.3810, 0.3991, 0.3660, 0.3725, 0.3639, 0.3896, 0.3821, 0.3695, 0.3863, |         |
|             | 0.3769, 0.3829, 0.3864, 0.3644, 0.3589,                      |  0.3755 |
| contingency | 0.5825, 0.5825, 0.5749, 0.5792, 0.5756, 0.5778, 0.5762, 0.5664, 0.5727, 0.5811, |         |
|             | 0.5775, 0.5840, 0.5691, 0.5771, 0.5780                       |  0.5770 |
| expansion   | 0.6807, 0.6847, 0.7027, 0.6892, 0.6821, 0.6944, 0.7017, 0.6978, 0.6806, 0.7002, |         |
|             | 0.6996, 0.6864, 0.6886, 0.6883, 0.6945                       |  0.6914 |
| expan+      | 0.8069, 0.8073, 0.8002, 0.8056, 0.8070, 0.8025, 0.8073, 0.7975, 0.8014, 0.8118, |         |
|             | 0.7987, 0.8054, 0.8067, 0.7970, 0.7944                       |  0.8033 |
| temporal    | 0.2571, 0.2675, 0.2796, 0.2683, 0.2656, 0.2564, 0.2727, 0.2740, 0.2795, 0.2658, |         |
|             | 0.3185, 0.2581, 0.2658, 0.2663, 0.2702                       |  0.2710 |



句子长调成50，双向lstm+dropout，只用因果向量试试，试试fine-tune