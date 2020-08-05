### paper

* Kipf & Welling (2016), Semi-Supervised Classification with Graph Convolutional Networks
* Defferrard et al. (NIPS 2016), Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering



#### 问题描述

**1. 目标： **学习图的信号/特征函数$G=(V,E)$;

**2. 节点特征：** 每个节点$i$均有其特征$x_i$,可以用矩阵$X_{N*D}$表示。其中$N$表示节点数，$D$表示每个节点的特征数;

**3. 图结构特征：**图结构上的信息可以用邻接矩阵$A$表示;

**4. 输出：**一个$Z_{N*F}$的矩阵，$F$表示新的特征数;



#### 解决思路

在问题描述中我们提到两类特征：**节点特征**和**图结构特征**。如果我们使用节点特征进行标签补全，那么完全就可以看作是一个标准的结构化特征分类问题，所以我们现在的挑战是如何使用图结构特征帮助训练。公式化的描述为：
$$
H^{l+1}=f(H^{(l)},A)
$$
其中$H^{(0)}=X$，$H^{(l)}=Z$，$L$表示层数。

**解决思路I：**

首先，我们可以使用一种简单的思路：
$$
f(H^{(l)},A)=\sigma(AH^{l}W^{l})
$$
这里$W$是上一层特征$H$的权重；$\sigma$是激活函数，此处可以使用简单的 $ReLU$ 。

这种思路是基于**节点特征与其所有邻居节点有关**的思想。邻接矩阵$A$与特征$H$相乘，等价于令某节点的邻居节点的特征相加。多层隐含层，表示我们想近似利用多层邻居的信息。

**这样的思路存在两大问题：**

* 如果节点不存在自连接（自身与自身有一条边），邻接矩阵$A$在对角线位置的值为0。但事实上在特征提取时，自身的信息非常重要；

* 邻接矩阵$A$没有被标准化，这在提取图特征时可能存在问题，比如邻居节点多的节点倾向于有更大的特征值；

**解决思路II:**

> **来源：参考文献[2]**

基于以上问题，可以提出改进的思路：
$$
f(H^{(l)},A)=\sigma(\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$

1. 解决问题1的方法是利用以下式子：
   $$
   \hat{A}=A+I
   $$

2. 解决问题2的方法是让邻接矩阵规范化，其中矩阵$D$的定义如下：

$$
\hat{D}_{ii}=\sum_j\hat{A}_{ij}
$$

矩阵$D$是一个对角线矩阵，其对角线位置上的值也就是相应节点的度。

$\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}$为对称标准化

上述式子也可以表示为：
$$
h^{{l+1}}_{v_i}=\sigma(\Sigma_j\frac{1}{c_{ij}}h^{(l)}_{v_j}W^{(l)})
$$
其中 $c_{ij}=\sqrt{d_id_j}$ ， $d_i$和$d_j$分别表示节点$i$和节点$j$的度。

在这种思路下，使用多层隐含层进行训练，我们就可以使用多层邻居的信息。

对于整个训练过程，我们只需要对各层的权重矩阵 ![W](https://www.zhihu.com/equation?tex=W) 进行梯度下降训练，而规范化后的邻接矩阵，在训练过程中实际上是个常量：
$$
\hat{A}=\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}}
$$

### Semi-Supervised Classification with Graph Convolutional Networks 论文详解

#### Introduction

本文提出了一种图卷积网络（graph covolutional networks, GCNs），该网络是传统卷积算法在图结构数据上的一个变体，可以直接用于处理图结构数据。从本质上讲，GCN 是谱图卷积（spectral graph convolution） 的**局部一阶近似**（localized first-order approximation）。GCN的另一个特点在于其模型规模会随图中边的数量的增长而线性增长。总的来说，GCN 可以用于对**局部**图结构与节点特征进行编码。

#### **Symbols**

- $\mathcal{G}=(\mathcal{V},\mathcal{E})$ 表示一个图， $\mathcal{V},\mathcal{E}$ 分别表示相应的节点集与边集，$u,v\in\mathcal{V}$ 表示图中的节点， $(u,v)\in\mathcal{E}$表示图中的边。
- $A$表示图的邻接矩阵（adjacency matrix）。
- $D$ 表示图的度矩阵（degree matrix）。
- $L$表示图的拉普拉斯矩阵（Laplacian matrix），$\mathcal{L}$表示图的归一化拉普拉斯矩阵。

#### Definition

拉普拉斯矩阵：
$$
\begin{equation}  
L(u,v)=
\left\{  
	\begin{array}{**l**}  
	d_v & if \space u=v \\  
	-1 & if \space (u,v) \space in \space \mathcal{E} \\  
	0 & othewise 
	\end{array}  
\right.  
\end{equation}
$$
其中，$d_v$表示节点$v$的度，则该矩阵称之为图$\mathcal{G}$的拉普拉斯矩阵$(L=D-A)$。相应的归一化拉普拉斯矩阵为：
$$
\begin{equation}  
L(u,v)=
\left\{  
	\begin{array}{**l**}  
	1 & if \space u=v \space and \space d_v \not=0 \\  
	-\frac{1}{\sqrt{d_u}\sqrt{d_v}} & if \space (u,v) \space in \space \mathcal{E} \\  
	0 & othewise 
	\end{array}  
\right.  
\end{equation}
$$
因此，图的归一化拉普拉斯矩阵可以通过下式计算：
$$
\mathcal{L}=D^{-\frac{1}{2}}LD^{\frac{1}{2}}=I-D^{-\frac{1}{2}}AD^{\frac{1}{2}}
$$

#### Motivation

**谱图卷积**

从本质上说，GCN是谱图卷积的一阶局部近似。那么，什么是谱图卷积呢？

首先来看图上的谱卷积。图上的谱卷积可以定义为信号 $x\in\mathbb{R}^{N}$ 与滤波器 $g_{\theta}=diag(\theta), \theta\in\mathbb{R}^{N}$ 在傅里叶域的乘积：
$$
\mathcal{g}_\theta\star x=U\mathcal{g}_\theta U^{T} x
$$
其中，$U$为归一化拉普拉斯 $L=I_{N}-D^{-\frac{1}{2}}AD^{-\frac{1}{2}}=U\Lambda U^{T}$的特征向量矩阵（即谱矩阵），其中， $\Lambda$为相应的特征值矩阵（对角矩阵）， $U^{T}x$为$x$的图傅氏变换。在这里，可以将 $g_{\theta}$看作是$L$特征向量的函数，也就是 $g_{\theta}(\Lambda)$。

对于图谱卷积来说，其计算代价无疑是很大的：(1) 使用$U$进行矩阵乘法运算的计算复杂度为 $\mathcal{O}(N^{2})$ ; (2)计算大图的拉普拉斯矩阵$L$的特征分解需要很大的计算量。针对这一问题，本文采用了[2]中的方法来近似 $g_{\theta}(\Lambda)$ 。该方法使用切比雪夫多项式（Chebyshev polynomial） $T_{k}(x)$的$K$阶截断来获得对 $g_{\theta}(\Lambda)$ 的近似：
$$
g_{\theta^{'}}(\Lambda)\approx\sum_{k=0}^{K}\theta_k^{'}T_k(\tilde{\Lambda})
$$
其中， $\tilde{\Lambda}=\frac{2}{\lambda_{max}}\Lambda-I_{N}$ 为经$L$的最大特征值（即谱半径）缩放后的特征向量矩阵。 $\theta'\in \mathbb{R}^{K}$ 表示一个切比雪夫向量。切比雪夫多项式使用递归的方式进行定义：$T_{k}(x)=2xT_{k-1}(x)-T_{k-2}(x)$，其中， $T_{0}(x)=1$且 $T_{1}(x)=x$。

此时，用近似的 $g_{\theta’}$ 替代原来的 $g_{\theta}$，可以得到：
$$
\mathcal{g}_{\theta^{'}}\star x=U\sum_{k=0}^{K}\theta_k^{'}T_k(\tilde{\Lambda}) U^{T} x=\sum_{k=0}^{K}\theta_k^{'}UT_k(\tilde{\Lambda})U^{T}x
$$
而 $T_{k}(\tilde{\Lambda})$ 是 $\Lambda$的 $k$阶多项式，且有 $U\tilde{\Lambda}^{k}U^{T}=(U\tilde{\Lambda}U^{T})^{k}=\tilde{L}^{k}$，其中， $\tilde{L}=\frac{2}{\lambda_{max}}L-I_{N}$ 。这样，我们就得到了文中的公式：
$$
\mathcal{g}_{\theta^{'}}\star x\approx \sum_{k=0}^{K}\theta_k^{'}T_k(\tilde{L})x
$$
通过这一近似，可以发现，**谱图卷积不再依赖于整个图，而只是依赖于距离中心节点 $K$步之内的节点（即$K$阶邻居）**。在[3]中，Defferrard et al. 使用了这一概念定义了图上的卷积神经网络。

#### Layer-wise线性模型

近似的谱图卷积虽然可以建立起$K$阶邻居的依赖，然而，却仍然需要在 $\tilde{L}$上进行$K$阶运算。在实际过程中，这一运算的代价也是非常大的。为了降低运算代价，本文进一步简化了该运算，即限定 $K=1$。此时，谱图卷积可以近似为 $\tilde{L}$（或$L$）的线性函数。

当然，这样做的代价是，只能建立起一阶邻居的依赖。对于这一问题，可以通过堆积多层图卷积网络建立$K$阶邻居的依赖，而且，这样做的另一个优势是，在建立 $K>1$阶邻居的依赖时，不需要受到切比雪夫多项式的限制。

为了进一步简化运算，在GCN的线性模型中，本文定义 $\lambda_{max}\approx 2$ 。此时，我们可以得到图谱卷积的一阶线性近似：
$$
g_{\theta^{'}}\star x \approx \theta^{'}_0 x+\theta^{'}_1(L-I_N)x=\theta^{'}_0 x-\theta^{'}_1D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$
可以看到，该式中仅有两个参数 $\theta_{0}’$ 与 $\theta_{1}'$。若需建立$k$阶邻居上的依赖，可以通过设置$k$层这样的滤波器来实现。

在实际的过程中，可以通过对参数进行约束来避免过拟合，并进一步简化运算复杂度。例如可以令 $\theta=\theta_{0}'=-\theta_{1}’$ ，从而得到
$$
g_{\theta^{'}}\star x \approx =\theta(I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
$$
需要注意的是， $I_{N}+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$ 的特征值范围为[0,2]，这意味着，当不停地重复该操作时（网络非常深时），可能会引起梯度爆炸或梯度消失。为了减弱这一问题，本文提出了一种 renormalization trick：
$$
I_N+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}\rightarrow \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}
$$
其中， $\tilde{A}=A+I_{N}$ ， $\tilde{D}_{ii}=\Sigma_{j}\tilde{A}_{ij}$。

当图中每个节点的表示不是单独的标量而是一个大小为$C$的向量时，可以使用其变体进行处理：
$$
Z=\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X\Theta
$$


其中， $\Theta\in\mathbb{R}^{C\times F}$表示参数矩阵， $Z\in\mathbb{R}^{N\times F}$为相应的卷积结果。此时，每个节点的节点表示被更新成了一个新的$F$维向量，该$F$维向量包含了相应的一阶邻居上的信息。

#### 图卷积神经网络

经过以上的推导，本文得到了图卷积神经网络的（单层）最终形式：
$$
H^{(l+1)}=\sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
$$
其中， 第$l$层网络的输入为 $H^{(l)}\in\mathbb{R}^{N\times D}$（初始输入为 $H^{(0)}=X$），$N$为图中的节点数量，每个节点使用$D$维的特征向量进行表示。 $\tilde{A}=A+I_{N}$为添加了自连接的邻接矩阵， $\tilde{D}$为度矩阵，$\tilde{D}_{ii}=\Sigma_{j}\tilde{A}_{ij}$。 $W^{(l)}\in \mathbb{R}^{D\times D}$为待训练的参数。 $\sigma$ 为相应的激活函数，例如 $ReLU(·)=max(0,·)$。此即GCN的最终形式。

#### Model

对于一个大图（例如“文献引用网络”），我们有时需要对其上的节点进行分类。然而，在该图上，仅有少量的节点是有标注的。此时，我们需要依靠这些已标注的节点来对那些没有标注过的节点进行分类，此即半监督节点分类问题。在这类问题中，由于大部分节点都没有已标注的标签，因此往往需要使用某种形式的图正则项对标签信息进行平滑（例如在损失函数中引入图拉普拉斯正则（graph Laplacian regularization））：
$$
\mathcal{L}=\mathcal{L}_0+\lambda\mathcal{L}_{reg},\space with \space \mathcal{L}_{reg}=\sum_{i,j}A_{ij}\|f(X_i)-f(X_j)\|^2=f(X)^{T}\Delta f(X)
$$


其中， $\mathcal{L}_{0}$ 表示有监督的损失， $f(·)$可以是一个类似于神经网络的可微函数。 $\lambda$表示一个权值因子， $X$则是相应的节点向量表示。 $\Delta=D-A$表示未归一化的图拉普拉斯矩阵。这种处理方式的一个基本假设是：**相连的节点可能有相同的标签**。然而，这种假设却往往会限制模型的表示能力，因为图中的边不仅仅可以用于编码节点相似度，而且还包含有额外的信息。

GCN的使用可以有效地避开这一问题。GCN通过一个简单的映射函数 $f(X,A)$ ，可以将节点的局部信息汇聚到该节点中，然后仅使用那些有标注的节点计算 $\mathcal{L}_{0}$即可，从而无需使用图拉普拉斯正则。

具体来说，本文使用了一个两层的GCN进行节点分类。模型结构图如下图所示：

![gcn](/Users/aszzy/Documents/study/note/pictures/models/gcn/model.png)

其具体流程为：

- 首先获取节点的特征表示$X$并计算邻接矩阵 $\hat{A}=\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$

- 将其输入到一个两层的GCN网络中，得到每个标签的预测结果：
  $$
  X=f(X,A)=softmax(\hat{A}\space ReLU(\hat{A}XW^{(0)})W^{(1)})
  $$



其中， $W^{(0)}\in\mathbb{R}^{C\times H}$为第一层的权值矩阵，用于将节点的特征表示映射为相应的隐层状态。 $W^{(1)}\in\mathbb{R}^{H\times F}$为第二层的权值矩阵，用于将节点的隐层表示映射为相应的输出（ $F$对应节点标签的数量）。最后将每个节点的表示通过一个softmax函数，即可得到每个标签的预测结果。

对于半监督分类问题，使用所有有标签节点上的期望交叉熵作为损失函数：
$$
\mathcal{L}=-\sum_{l \in \mathcal{y}_L}\sum^{F}_{f=1}Y_{lf}lnZ_{lf}
$$
其中， $\mathcal{Y}_{L}$表示有标签的节点集。

#### Experiments

针对半监督节点分类问题，本文主要进行了两个实验：一是在文献引用网络上的实验，二是在知识图谱上的实验（NELL）。在文献引用网络中，边使用引用链构建，节点表示相应的文档。本文共使用了三个引用网络数据集：Citeseer、Cora与Pubmed。其数据统计的结果如下表所示（Label rate表示有标注节点的比例）：

![gcn](/Users/aszzy/Documents/study/note/pictures/models/gcn/data.png)

在这些数据集上的节点分类实验结果如下表所示：

![gcn](/Users/aszzy/Documents/study/note/pictures/models/gcn/accuracy.png)

文献引用试验结果：

![gcn](/Users/aszzy/Documents/study/note/pictures/models/gcn/comparison.png)

#### Conclusion

本文提出了一种图卷积神经网络，该网络可以被有效地用于处理图结构的数据。图卷积神经网络具有几个特点：

- 局部特性：图卷积神经网络关注的是图中以某节点为中心，K阶邻居之内的信息，这一点与GNN有本质的区别；
- 一阶特性：经过多种近似之后，GCN变成了一个一阶模型。也就是说，单层的GCN可以被用于处理图中一阶邻居上的信息；若要处理K阶邻居，可以采用多层GCN来实现；
- 参数共享：对于每个节点，其上的滤波器参数$W$是共享的，这也是其被称作图卷积网络的原因之一。

总的来说，图卷积神经网络是神经网络技术在图结构数据上的一个重要应用。目前，已经出现了很多GCN的变体，这也反映了如何将神经网络与图结构数据结合起来，已经成为了目前的一个热点问题。这一研究可以被广泛应用在graph embedding、node embedding等图相关的任务上，它也为处理大型图结构数据提供了一种有效的手段。