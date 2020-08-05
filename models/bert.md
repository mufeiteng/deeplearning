### BERT

三种模型中，只有BERT表征会基于所有层中的左右两侧语境。

- BERT 使用双向Transformer
- OpenAI GPT 使用从左到右的Transformer
- ELMo 使用独立训练的从左到右和从右到左LSTM的级联来生成下游任务的特征。

![bert-gpt-transformer-elmo](/Users/ftmu/Documents/study/deep_learning/pictures/models/bert/bert-gpt-transformer-elmo.png)

### Input Representation

论文的输入表示（input representation）能够在一个token序列中明确地表示单个文本句子或一对文本句子（例如， [Question, Answer]）。对于给定token，其输入表示通过对相应的token、segment和position embeddings进行求和来构造：

![bert-input-representation](/Users/ftmu/Documents/study/deep_learning/pictures/models/bert/bert-input-representation.png)

- 使用WordPiece嵌入【GNMT，[Google’s neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1609.08144)】和30,000个token的词汇表。用##表示分词。
- 使用learned positional embeddings，支持的序列长度最多为512个token。
- 每个序列的第一个token始终是特殊分类嵌入（[CLS]）。对应于该token的最终隐藏状态（即，Transformer的输出）被用作分类任务的聚合序列表示。对于非分类任务，将忽略此向量。
- 句子对被打包成一个序列。以两种方式区分句子。
  - 首先，用特殊标记（[SEP]）将它们分开。
  - 其次，添加一个learned sentence A嵌入到第一个句子的每个token中，一个sentence B嵌入到第二个句子的每个token中。
- 对于单个句子输入，只使用 sentence A嵌入。

### Pre-training Tasks

- 它在训练双向语言模型时以减小的概率把少量的词替成了Mask或者另一个随机的词。感觉其目的在于使模型被迫增加对上下文的记忆。（知乎的回答）
- 增加了一个预测下一句的loss。

#### Task #1: Masked LM

标准条件语言模型只能从左到右或从右到左进行训练，因为双向条件作用将允许每个单词在多层上下文中间接地“see itself”。

为了训练一个深度双向表示（deep bidirectional representation），研究团队采用了一种简单的方法，即随机屏蔽（masking）部分输入token，然后只预测那些被屏蔽的token。论文将这个过程称为“masked LM”(MLM)，尽管在文献中它经常被称为Cloze任务(Taylor, 1953)。

在这个例子中，与masked token对应的最终隐藏向量被输入到词汇表上的输出softmax中，就像在标准LM中一样。在团队所有实验中，随机地屏蔽了每个序列中15%的WordPiece token。与去噪的自动编码器（Vincent et al.， 2008）相反，只预测masked words而不是重建整个输入。

虽然这确实能让团队获得双向预训练模型，但这种方法有两个缺点。

- 缺点1：预训练和finetuning之间不匹配，因为在finetuning期间从未看到`[MASK]`token。

为了解决这个问题，团队并不总是用实际的`[MASK]`token替换被“masked”的词汇。相反，训练数据生成器随机选择15％的token。

例如在这个句子“my dog is hairy”中，它选择的token是“hairy”。然后，执行以下过程：

数据生成器将执行以下操作，而不是始终用`[MASK]`替换所选单词：

- 80％的情况：用`[MASK]`标记替换单词，例如，`my dog is hairy → my dog is [MASK]`
- 10％的情况：用一个随机的单词替换该单词，例如，`my dog is hairy → my dog is apple`
- 10％的情况：保持单词不变，例如，`my dog is hairy → my dog is hairy`. 这样做的目的是将表示偏向于实际观察到的单词。

Transformer encoder不知道它将被要求预测哪些单词或哪些单词已被随机单词替换，因此它被迫保持每个输入token的分布式上下文表示。此外，因为随机替换只发生在所有token的1.5％（即15％的10％），这似乎不会损害模型的语言理解能力。

- 缺点2：每个batch只预测了15％的token，这表明模型可能需要更多的预训练步骤才能收敛。

团队证明MLM的收敛速度略慢于 left-to-right的模型（预测每个token），但MLM模型在实验上获得的提升远远超过增加的训练成本。

#### Task #2: Next Sentence Prediction

在为了训练一个理解句子的模型关系，预先训练一个二分类的下一句测任务，这一任务可以从任何单语语料库中生成。具体地说，当选择句子A和B作为预训练样本时，B有50％的可能是A的下一个句子，也有50％的可能是来自语料库的随机句子。例如：

```python
Input = 
[CLS] the man went to [MASK] store [SEP]
he bought a gallon [MASK] milk [SEP]
Label = IsNext
 
Input = 
[CLS] the man [MASK] to the store [SEP]
penguin [MASK] are flight ##less birds [SEP]
Label = NotNext

```

完全随机地选择了NotNext语句，最终的预训练模型在此任务上实现了97％-98％的准确率。

