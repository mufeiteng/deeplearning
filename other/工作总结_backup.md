[TOC]

---

#### Why-QA

##### 数据集: Yahoo QA数据集

通过"What causes ..." 以及 "What is the result of ..."抽取了3031个question,每个question有至少四个候选答案,五折交叉验证

##### 难点

可能也存在数据集太小的问题

##### 做法

* 原因结果二分类+句对分类模型
* 加入额外特征
* 加入因果向量



#### 因果推理

##### 数据集：COPA

1. dev、test set各500个, 例子为:

```c
1. My body cast a shadow over the grass. What was the CAUSE of this?
       The sun was rising.
       The grass was cut.
2. The woman repaired her faucet. What was the CAUSE of this?
       The faucet was leaky.
       The faucet was turned off.

```

形式为句对建模问题, 但缺少训练集,只有测试集和验证集.

1. 句子较短,一般不超过10个词,句式较为简单,基本为主谓宾结构. domain偏向日常生活,如上例.
2. 候选为两个,随机选择的accuracy为0.5
3. 对数据集分析:因果关系较多的存在于词级别, 如<sun, shadow> <leaky, repaired>
4. **state-of-art的做法**: 使用10TB的语料抽取关系对,在单词的层面计算词对的PMI值,对句对进行因果打分,选取打分高的候选;因为语料很大,基本能够cover 测试集,所以取得了较好的结果
5. 目前还没有基于神经网络的做法

##### 难点

- 没有训练集,只能通过人工抽取的数据进行训练,所以语料domain不匹配的问题可能存在.
- 由于偏向日常生活,所以使用传记,小说类型的语料可能取得好一点的结果

##### 词表cover情况

1. 测试集

- 验证集词表:原因词1377个, 结果词1307个.
- 因果向量词表包含:原因词1222个,结果词1072个.

2. 验证集

- 验证集词表:原因词1437个, 结果词1334个.
- 因果向量词表包含:原因词1266个,结果词1130个.

##### 做法

- 放低对pattern精度的要求,抽取更多的语料,提升覆盖度
- sharp公布的数据集有80w+个句对,缺少功能词.但是在这个问题中功能词并不是很重要,可以把这部分数据利用起来训练分类器



#### 因果向量

##### 改进负采样

1. 随机采样的问题

   - 可能会采样到因果词对

   - 采样的句对太简单，没有区分度

2. 做法如下

- 预训练Denoising Auto-encoder（DAE）
- 随机选取因果句对$<C,E>$ ，记为$S_{real}$，进行单词的随机删除或者替换，得到$<C_{noise},E_{noise}>$
- DAE以 $<C_{noise},E_{noise}>$ 为输入，生成负样本$S_{fake}$
- discriminator(D):

$$
-log\space p(D(S_{real}))-log \space (1-p(D(S_{fake})))
$$

- generator(G):

$$
-log \space (p(D(S_{fake})))
$$



#### Pattern表示

##### 提高cover度

1. 现有high precision线索词cover不高，抽取出的数据很稀疏

2. 因果动词可以穷举，但是其中很多(记为$Cue_{blur}$)精度不高，需要对其质量打分

- 使用high precision因果句对$pair_{hp}$训练分类器$Classifier$
- 使用$Cue_{blur}$抽取短语对$pair_{blur}$ ，使用$Classifier$对$pair_{blur}$（$Cue_{blur}$）打分，分值大于阈值$\rho$的记为$pair_{hp}$
- 用测试数据对$pair_{hp}$+$pair_{hp}$ 评估，保证accuracy情况下，调整$\rho$

3. 使用$Cue_{blur}$抽取 SVO三元组，设定阈值，用测试数据对其打分，调整阈值，不断迭代

##### 识别pattern

1. 产生候选：对所有动词，抽取SVO三元组（事件表示）
2. 用所有数据训练分类器，对SVO打分，分值大于$\alpha$ 的记为$SVO_{retain}$ 
3. 用测试数据评估 $SVO_{retain}$ ，调整$\alpha$ ，不断迭代



#### 隐式篇章

##### 数据集：PDTB

| relation    | train | dev  | test |
| ----------- | ----- | ---- | ---- |
| Comparison  | 1942  | 197  | 152  |
| Contingency | 3342  | 295  | 279  |
| Expansion   | 7004  | 671  | 574  |
| Temporal    | 760   | 64   | 85   |

##### 难点

1. 语料为Wall Street Journal,与从英文wiki的抽取的关系的domain可能不一样
2. 数据集太小,容易过拟合,训练过程很不稳定
3. 各论文并没有明确的motivation，主体思路为尝试不同的attention,例如: **为了更好的建模句子表示及句对信息交互,本文提出了一种新的attention机制,并获得了state-of-art结果**
4. 最简单的CNN句对模型（不加attention）便能取得较好结果，然后加component便会过拟合
5. 关系的线索词,重合度很大,同一个句对可能属于多种关系.

##### 关系抽取

1. 书写规则,对四种关系进行了抽取,数量如下

| 关系类型    | 抽取数量 | pattern数量 |
| ----------- | -------- | ----------- |
| contingency | 330713   | 很多        |
| expansion   | 31590    | 9个         |
| temporal    | 34072    | 8个         |
| comparison  | 82036    | 11个        |

2. 可能的问题

* 各关系的线索词集较小，可能存在对应的PDTB关系cover不够的问题
* 由于要确保pattern的质量，抽取的数量较少

3. 词表cover情况
   * **验证集**: 原因2195  结果1935 **因果向量cover**: 原因1958  结果1661
   * **测试集**: 原因1761  结果1651 **因果向量cover**: 原因1539  结果1445

##### 时态特征





==单个句子时态种类分布，句对时态组合分布==

##### 做过的尝试

1. 多任务：区分内容词，功能词 
2. 加外部数据
3. 单任务分类：$Classifier_{arg1}$ + $Classifier_{arg2}$  + $Classifier_{<arg1,arg2>}$ 

==结果怎样，能得到什么结论==

##### 可能的方向

==加强因果向量,注意句法特征,尝试生成模型,==





#### 通用词向量

##### 各个数据集的特点

##### 可能的方向

==GAN根据synthetic上下文生成负样本==





#### SVO三元组

**目的**：提升recall

1. 现有bk,sg的因果词对测试数据$Test_{word}$，以及2000条评估抽取质量的标注数据$Test_{noise}$，可以构建0/1因果句对测试数据$Test_{sentence}$  

2. 保证因果向量的质量下，抽取更多的因果pair：

   1）low precision的线索词$Cue_{blur}$  

   2）使用high precision因果句对$pair_{hp}$ 训练分类器$Classifier $

   3）使用$Cue_{blur}$抽取短语对$pair_{blur}$，使用$Classifier$对$pair_{blur}$（$Cue_{blur}$）打分，分值大于阈值$\rho$的记为$pair_{new} $

   4）用$Test_{word}$和$Test_{sentence}$评估$pair_{new}$,保证质量的情况下，调整$\rho$  

   **最终获得较高cover的因果线索词和短语对**$pair_{new}$

3. 使用$Cue_{blur}$ 抽取SVO三元组$causal_{SVO}$ ，用2得到的$pair_{new}$训练分类器 ,对$causal_{SVO}$打分,设定阈值，将打分较高的取出，用$Test_{sentence}$ 评估质量，通过调整阈值，获得质量较高的$causal_{SVO}$ 

   **不断迭代**



#### 功能词内容词



频繁模式







哪几个方向可以做，分析难点在哪

通用词向量，各个数据集的区别

怎么建模pattern的表示，如何实施



隐式篇章PDTB，Why-QA，Copa，数据集的特点，怎么建模





词的cover，copa，Yahoo 都覆盖了多少

其他的关系的建模，





