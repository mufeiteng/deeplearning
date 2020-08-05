残差网络

图卷积

bert

The Shattered Gradients Problem: If resnets are the answer, then what is the question?



### Attention is all your need

##### 特点

* CNN 编码局部信息，没有全局上下文
* RNN编码全局信息，不能并行
* Transformer能并行，编码全局信息

##### Attention定义

<img src='/Users/ftmu/Documents/study/note/pictures/models/transformer/attention.png' height='200px'>

transformer就是一个encoder，用来编码序列
$$
Attention(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = softmax\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d_k}}\right)\boldsymbol{V}
$$
其中$\boldsymbol{Q}\in\mathbb{R}^{n\times d_k}, \boldsymbol{K}\in\mathbb{R}^{m\times d_k}, \boldsymbol{V}\in\mathbb{R}^{m\times d_v}$ , 于是我们可以认为：这是一个Attention层，将$n×d_k$的序列Q编码成了一个新的$n×d_v$的序列, 就是做了一个矩阵变换，然后做了softmax。

##### Multi-Head Attention

<img src='/Users/ftmu/Documents/study/note/pictures/models/piecemeal/transformer.png' height='200px'>

![image-20200105180015219](/Users/ftmu/Library/Application Support/typora-user-images/image-20200105180015219.png)

就是把$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 通过参数矩阵映射一下，然后再做Attention，把这个过程重复做h次，结果拼接起来就行了，可谓“大道至简”了。

**所谓“多头”（Multi-Head），就是只多做几次同样的事情（参数不共享），然后把结果拼接**。
$$
head_i = Attention(\boldsymbol{Q}\boldsymbol{W}_i^Q,\boldsymbol{K}\boldsymbol{W}_i^K,\boldsymbol{V}\boldsymbol{W}_i^V)\\
MultiHead(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}) = Concat(head_1,...,head_h)
$$

##### Self Attention

到目前为止，对Attention层的描述都是一般化的，我们可以落实一些应用。比如，如果做阅读理解的话，Q可以是篇章的词向量序列，取K=V为问题的词向量序列，那么输出就是所谓的Aligned Question Embedding。

而在Google的论文中，大部分的Attention都是Self Attention，即“自注意力”，或者叫内部注意力。

所谓Self Attention，其实就是$Attention(X,X,X)$，X就是前面说的输入序列。也就是说，在序列内部做Attention，寻找序列内部的联系。Google论文的主要贡献之一是它表明了==内部注意力在机器翻译（甚至是一般的Seq2Seq任务）的序列编码上是相当重要的，而之前关于Seq2Seq的研究基本都只是把注意力机制用在解码端==。类似的事情是，目前SQUAD阅读理解的榜首模型R-Net也加入了自注意力机制，这也使得它的模型有所提升。

##### Position Embedding

这就表明了，到目前为止，Attention模型顶多是一个非常精妙的“词袋模型”而已。

这问题就比较严重了，大家知道，对于时间序列来说，尤其是对于NLP中的任务来说，顺序是很重要的信息，它代表着局部甚至是全局的结构，学习不到顺序信息，那么效果将会大打折扣（比如机器翻译中，有可能只把每个词都翻译出来了，但是不能组织成合理的句子）。

于是Google再祭出了一招——**Position Embedding**，也就是“位置向量”，将每个位置编号，然后每个编号对应一个向量，通过结合位置向量和词向量，就给每个词都引入了一定的位置信息，这样Attention就可以分辨出不同位置的词了。

结合位置向量和词向量有几个可选方案，可以把它们拼接起来作为一个新向量，也可以把位置向量定义为跟词向量一样大小，然后两者加起来。FaceBook的论文和Google论文中用的都是后者。直觉上相加会导致信息损失，似乎不可取，但Google的成果说明相加也是很好的方案。看来我理解还不够深刻。

### ResNet

##### 问题

==随着网络层级的不断增加，模型精度不断得到提升，而当网络层级增加到一定的数目以后，训练精度和测试精度迅速下降，这说明当网络变得很深以后，深度网络就变得更加难以训练了==。

<font color=#FF0000>为什么随着网络层级越深，模型效果却变差了呢？</font>

神经网络在反向传播过程中要不断地传播梯度，而当网络层数加深时，梯度在传播过程中会逐渐消失（假如采用Sigmoid函数，对于幅度为1的信号，每向后传递一层，梯度就衰减为原来的0.25，层数越多，衰减越厉害），导致无法对前面网络层的权重进行有效的调整。

那么，如何又能加深网络层数、又能解决梯度消失问题、又能提升模型精度呢？

##### idea

前面描述了一个实验结果现象，在不断加神经网络的深度时，模型准确率会先上升然后达到饱和，再持续增加深度时则会导致准确率下降。

那么我们作这样一个假设：<font color='#ff0000'>假设现有一个比较浅的网络（Shallow Net）已达到了饱和的准确率，这时在它后面再加上几个恒等映射层（Identity mapping，也即y=x，输出等于输入），这样就增加了网络的深度，并且起码误差不会增加，也即更深的网络不应该带来训练集上误差的上升。而这里提到的使用恒等映射直接将前一层输出传到后面的思想，便是著名深度残差网络ResNet的灵感来源。</font>

ResNet引入了残差网络结构（residual network），通过这种残差网络结构，可以把网络层弄的很深（据说目前可以达到1000多层），并且最终的分类效果也非常好，残差网络的基本结构如下图所示，很明显，该图是带有跳跃结构的：

<img src='/Users/aszzy/Documents/study/note/pictures/models/piecemeal/resnet.png' height='200px'>

- 实线的Connection部分，表示通道相同，如上图的第一个粉色矩形和第三个粉色矩形，都是3x3x64的特征图，由于通道相同，所以采用计算方式为H(x)=F(x)+x
- 虚线的的Connection部分，表示通道不同，如上图的第一个绿色矩形和第三个绿色矩形，分别是3x3x64和3x3x128的特征图，通道不同，采用的计算方式为H(x)=F(x)+Wx，其中W是卷积操作，用来调整x维度的。

残差网络借鉴了高速网络（Highway Network）的跨层链接思想，但对其进行改进（残差项原本是带权值的，但ResNet用恒等映射代替之）。
假定某段神经网络的输入是x，期望输出是H(x)，即H(x)是期望的复杂潜在映射，如果是要学习这样的模型，则训练难度会比较大；
回想前面的假设，如果已经学习到较饱和的准确率（或者当发现下层的误差变大时），那么接下来的学习目标就转变为恒等映射的学习，也就是使输入x近似于输出H(x)，以保持在后面的层次中不会造成精度下降。
在上图的残差网络结构图中，通过“shortcut connections（捷径连接）”的方式，==直接把输入x传到输出作为初始结果，输出结果为H(x)=F(x)+x，当F(x)=0时，那么H(x)=x，也就是上面所提到的恒等映射。于是，ResNet相当于将学习目标改变了，不再是学习一个完整的输出，而是目标值H(X)和x的差值，也就是所谓的残差F(x) := H(x)-x，因此，后面的训练目标就是要将残差结果逼近于0，使到随着网络加深，准确率不下降。==
这种残差跳跃式的结构，打破了传统的神经网络n-1层的输出只能给n层作为输入的惯例，使某一层的输出可以直接跨过几层作为后面某一层的输入，其意义在于为叠加多层网络而使得整个学习模型的错误率不降反升的难题提供了新的方向。

### Google-BERT

