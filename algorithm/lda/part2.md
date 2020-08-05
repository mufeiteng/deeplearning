#### LDA 模型

##### PLSA 和 LDA 的区别

首先，我们来看看PLSA和LDA生成文档的方式。在PLSA中，生成文档的方式如下：

- 按照概率$p(d_i)$选择一篇文档$d_i$
- 根据选择的文档$d_i$，从主题分布中按照概率$p(\zeta_k \mid d_i)$选择一个隐含的主题类别$\zeta_k$
- 根据选择的主题$\zeta_k$, 从词分布中按照概率$p(\omega_j \mid \zeta_k)$选择一个词$\omega_j$

LDA 中，生成文档的过程如下：

1.上帝有两大坛骰子，都一个坛子装的是doc-topic骰子，第二个坛子装的是topic-word骰子；

2.上帝随机的从第二坛骰子中独立的抽取了K个topic-word骰子，编号为1到K；

3.每次生成一篇新的文档前，上帝先从第一个坛子里随机抽取一个doc-topic骰子。然后重复以下过程生成文档中的词

投掷这个topic-word骰子，得到一个topic编号z, 选择K个topic-word骰子中编号为z的那个，投掷这个骰子，于是得到一个词



- **按照先验概率 $p(d_i)$ 选择一篇文档 $d_i$** 
- **从Dirichlet分布** $\alpha$ **中取样生成文档** $d_i$ **的主题分布** $\theta_i$ **，主题分布** $\theta_i$ **由超参数为** $\alpha$ **的Dirichlet分布生成**
- **从主题的多项式分布** $\theta_i$ **中取样生成文档** $d_i$ **第 j 个词的主题** $z_{i, j}$ 
- **从Dirichlet分布** $\beta$ **中取样生成主题** $z_{i, j}$ **对应的词语分布** $\phi_{z_{i, j}}$ **，词语分布** $\phi_{z_{i, j}}$ **由参数为** $\beta$ **的Dirichlet分布生成**
- **从词语的多项式分布** $\phi_{z_{i, j}}$ **中采样最终生成词语** $\omega_{i, j}$   

可以看出，LDA 在 PLSA 的基础上，为主题分布和词分布分别加了两个 Dirichlet 先验。

举个例子，如图所示：

<img src="/Users/aszzy/Documents/study/note/pictures/algorithm/1.jpg" height="200px" />

上图中有三个主题，在PLSA中，我们会以固定的概率来抽取一个主题词，比如0.5的概率抽取教育这个主题词，然后根据抽取出来的主题词，找其对应的词分布，再根据词分布，抽取一个词汇。由此，**可以看出PLSA中，主题分布和词分布都是唯一确定的。但是，在LDA中，主题分布和词分布是不确定的，LDA的作者们采用的是贝叶斯派的思想，认为它们应该服从一个分布，主题分布和词分布都是多项式分布，因为多项式分布和狄利克雷分布是共轭结构，在LDA中主题分布和词分布使用了Dirichlet分布作为它们的共轭先验分布。**所以，也就有了一句广为流传的话 ： LDA 就是 PLSA 的贝叶斯化版本。

<img src="/Users/aszzy/Documents/study/note/pictures/algorithm/2.png" height="200px" />



现在我们来详细讲解论文中的LDA模型，即上图。

$\vec \alpha \to \vec \theta_m \to \zeta_{m, n}$  , 这个过程表示在生成第m篇文档的时候，先从抽取了一个doc-topic骰子 $\vec \theta_m$ , 然后投掷这个骰子生成了文档中第n个词的topic编号 $\zeta_{m, n}$ ;

$\vec \beta \to \vec \phi_k \to \omega_{m, n}\mid = \zeta_{m ,n}$ , 这个过程表示，从K个topic-word骰子 $\vec \phi_k$ 中，挑选编号为 $k = \zeta_{m, n}$ 的骰子进行投掷，然后生成词汇 $\omega_{m , n}$ ;

在LDA中，也是采用词袋模型，M篇文档会对应M个独立Dirichlet-Multinomial共轭结构；K个topic会对应K个独立的Dirichlet-Multinomial共轭结构。

上面的LDA的处理过程是一篇文档一篇文档的过程来处理，并不是实际的处理过程。文档中每个词的生成都要抛两次骰子，第一次抛一个doc-topic骰子得到 topic, 第二次抛一个topic-word骰子得到 word，每次生成每篇文档中的一个词的时候这两次抛骰子的动作是紧邻轮换进行的。如果语料中一共有 N 个词，则上帝一共要抛 2N次骰子，轮换的抛doc-topic骰子和 topic-word骰子。但实际上有一些抛骰子的顺序是可以交换的，我们可以等价的调整2N次抛骰子的次序：前N次只抛doc-topic骰子得到语料中所有词的 topics,然后基于得到的每个词的 topic 编号，后N次只抛topic-word骰子生成 N 个word。此时，可以得到：
$$
\begin{align} p(\vec w , \vec z \mid \vec \alpha, \vec \beta) & = p(\vec w \mid \vec z, \vec \beta) p(\vec z \mid \vec \alpha) \\ & = \prod_{k=1}^K\frac{\Delta(\vec \phi_K + \vec \beta)}{\Delta (\vec \beta)} \prod_{m=1}^M \frac{\Delta(\vec \theta_m + \vec \alpha)}{\vec \alpha} \end{align}
$$



##### 另外的理解

如下图所示

<img src="/Users/aszzy/Documents/study/note/pictures/algorithm/lda.png" height="200px"></img>

LDA假设文档主题的先验分布是Dirichlet分布，即对于任一文档$d$, 其主题分布$\theta_d$为：$\theta_d = Dirichlet(\vec \alpha)$, 其中，$\alpha $ 为分布的超参数，是一个K维向量。

LDA假设主题中词的先验分布是Dirichlet分布，即对于任一主题k, 其词分布$\beta_k$为 $\beta_k= Dirichlet(\vec \eta)$ , 其中，$\eta$为分布的超参数，是一个V维向量, V代表词汇表里所有词的个数。

对于数据中任一一篇文档d中的第n个词，我们可以从主题分布$θ_d$中得到它的主题编号$z_{dn}$的分布为：
$$
z_{dn} = multi(\theta_d)
$$
而对于该主题编号，得到我们看到的词$w_{dn}$的概率分布为： 
$$
w_{dn} = multi(\beta_{z_{dn}})
$$
理解LDA主题模型的主要任务就是理解上面的这个模型。这个模型里，我们有M个文档主题的Dirichlet分布，而对应的数据有M个主题编号的多项分布，这样($α→θ_d→\vec{z}_{d}$)就组成了Dirichlet-multi共轭，可以使用前面提到的贝叶斯推断的方法得到基于Dirichlet分布的文档主题后验分布。

如果在第d个文档中，第k个主题的词的个数为：$n^{(k)}_d$, 则对应的多项分布的计数可以表示为
$$
\vec n_d = (n_d^{(1)}, n_d^{(2)},...n_d^{(K)})
$$
利用Dirichlet-multi共轭，得到$θ_d$的后验分布为 $Dirichlet(\theta_d | \vec \alpha + \vec n_d)$ .

同样的道理，对于主题与词的分布，我们有K个主题与词的Dirichlet分布，而对应的数据有K个主题编号的多项分布，这样($\eta→β_k→\vec{w}_{(k)}$)就组成了Dirichlet-multi共轭，可以使用前面提到的贝叶斯推断的方法得到基于Dirichlet分布的主题词的后验分布。

如果在第k个主题中，第v个词的个数为：$n^{(v)}_k$, 则对应的多项分布的计数可以表示为
$$
\vec n_k = (n_k^{(1)}, n_k^{(2)},...n_k^{(V)})
$$
利用Dirichlet-multi共轭，得到$β_k$的后验分布为：$Dirichlet(\beta_k | \vec \eta+ \vec n_k)$ .

##### 使用Gibbs Sampling进行采样

在Gibbs采样算法求解LDA的方法中，我们的$α,η$是已知的先验输入,我们的目标是得到各个$z_{dn},w_{kn}$对应的整体$\vec{z},\vec{w}$的概率分布，即文档主题的分布和主题词的分布。由于我们是采用Gibbs采样法，则对于要求的目标分布，我们需要得到对应分布各个特征维度的条件概率分布。

具体到我们的问题，我们的所有文档联合起来形成的词向量$\vec{w}$是已知的数据，不知道的是语料库主题$\vec{z}$的分布。假如我们可以先求出$w,z$的联合分布$p(\vec{w},\vec{z})$，进而可以求出某一个词$w_i$对应主题特征$z_i$的条件概率分布$p(z_i=k|\vec{w},\vec z_{\neg i})$。其中，$\vec z_{\neg i}$代表去掉下标为i的词后的主题分布。有了条件概率分布$p(z_i=k|\vec{w},\vec z_{\neg i})$，我们就可以进行Gibbs采样，最终在Gibbs采样收敛后得到第i个词的主题。

如果我们通过采样得到了所有词的主题,那么通过统计所有词的主题计数，就可以得到各个主题的词分布。接着统计各个文档对应词的主题计数，就可以得到各个文档的主题分布。

以上就是Gibbs采样算法求解LDA的思路。

##### 主题和词的联合分布与条件分布的求解

从上一节可以发现，要使用Gibbs采样求解LDA，关键是得到条件概率$p(z_i=k| \vec w,\vec z_{\neg i})$的表达式。那么这一节我们的目标就是求出这个表达式供Gibbs采样使用。

首先我们简化下Dirichlet分布的表达式,其中$\triangle(\alpha)$是归一化参数：
$$
Dirichlet(\vec p| \vec \alpha) = \frac{\Gamma(\sum\limits_{k=1}^K\alpha_k)}{\prod_{k=1}^K\Gamma(\alpha_k)}\prod_{k=1}^Kp_k^{\alpha_k-1} = \frac{1}{\triangle( \vec \alpha)}\prod_{k=1}^Kp_k^{\alpha_k-1}
$$
现在我们先计算下第d个文档的主题的条件分布$p(\vec z_d|\alpha)$，在上一篇中我们讲到$\alpha \to \theta_d \to \vec z_d$组成了Dirichlet-multi共轭,利用这组分布，计算$p(\vec z_d| \vec \alpha)$如下：
$$
\begin{align}\begin{split} p(\vec z_d| \vec \alpha)  & = \int p(\vec z_d |  \vec \theta_d) p(\theta_d |  \vec \alpha) d  \vec \theta_d \\ & = \int \prod_{k=1}^Kp_k^{n_d^{(k)}} Dirichlet(\vec \alpha) d \vec \theta_d \\ & = \int \prod_{k=1}^Kp_k^{n_d^{(k)}} \frac{1}{\triangle( \vec \alpha)}\prod_{k=1}^Kp_k^{\alpha_k-1}d \vec \theta_d \\ & =  \frac{1}{\triangle( \vec \alpha)} \int \prod_{k=1}^Kp_k^{n_d^{(k)} + \alpha_k-1}d \vec \theta_d \\ & = \frac{\triangle(\vec n_d +  \vec \alpha)}{\triangle( \vec \alpha)} \end{split}  \end{align}
$$
其中，在第d个文档中，第k个主题的词的个数表示为：$n_d^{(k)}$ , 对应的多项分布的计数可以表示为
$$
\vec n_d = (n_d^{(1)}, n_d^{(2)},...n_d^{(K)})
$$
有了单一一个文档的主题条件分布，则可以得到所有文档的主题条件分布为：
$$
p(\vec z|\vec \alpha) =  \prod_{d=1}^Mp(\vec z_d|\vec \alpha) =  \prod_{d=1}^M \frac{\triangle(\vec n_d +  \vec \alpha)}{\triangle( \vec \alpha)}
$$
同样的方法，可以得到，第k个主题对应的词的条件分布$p(\vec w|\vec z, \vec \eta)$为：
$$
p(\vec w|\vec z, \vec \eta) =\prod_{k=1}^Kp(\vec w_k|\vec z, \vec \eta) =\prod_{k=1}^K \frac{\triangle(\vec n_k +  \vec \eta)}{\triangle( \vec \eta)}
$$
其中，第k个主题中，第v个词的个数表示为：$n^{(v)}_k$, 对应的多项分布的计数可以表示为
$$
\vec n_k = (n_k^{(1)}, n_k^{(2)},...n_k^{(V)})
$$

最终我们得到主题和词的联合分布$p(\vec w, \vec z| \vec \alpha,  \vec \eta)$如下：
$$
p(\vec w, \vec z)  \propto p(\vec w, \vec z| \vec \alpha,  \vec \eta) = p(\vec z|\vec \alpha) p(\vec w|\vec z, \vec \eta) =  \prod_{d=1}^M \frac{\triangle(\vec n_d +  \vec \alpha)}{\triangle( \vec \alpha)}\prod_{k=1}^K \frac{\triangle(\vec n_k +  \vec \eta)}{\triangle( \vec \eta)}
$$
有了联合分布，现在我们就可以求Gibbs采样需要的条件分布$p(z_i=k| \vec w,\vec z_{\neg i})$了。
需要注意的是这里的i是一个二维下标，对应第d篇文档的第n个词。对于下标i,由于它对应的词wi是可以观察到的，因此我们有：
$$
p(z_i=k| \vec w,\vec z_{\neg i}) \propto p(z_i=k, w_i =t| \vec w_{\neg i},\vec z_{\neg i})
$$
对于$z_i=k, w_i =t$,它只涉及到第d篇文档和第k个主题两个Dirichlet-multi共轭，即：
$$
\begin{align}\begin{split}
\vec \alpha \to \vec \theta_d \to \vec z_d\\
\vec \eta \to \vec \beta_k \to \vec w_{(k)}
\end{split}\end{align}
$$
其余的M+K−2个Dirichlet-multi共轭和它们这两个共轭是独立的。如果我们在语料库中去掉$z_i,w_i$,并不会改变之前的M+K个Dirichlet-multi共轭结构，只是向量的某些位置的计数会减少，因此对于$\vec \theta_d, \vec \beta_k$,对应的后验分布为：
$$
\begin{align}\begin{split}
p(\vec \theta_d | \vec w_{\neg i},\vec z_{\neg i}) = Dirichlet(\vec \theta_d | \vec n_{d, \neg i} + \vec \alpha)\\
p(\vec \beta_k | \vec w_{\neg i},\vec z_{\neg i}) = Dirichlet(\vec \beta_k | \vec n_{k, \neg i} + \vec \eta)
\end{split}\end{align}
$$
现在开始计算Gibbs采样需要的条件概率：
$$
\begin{align}\begin{split} p(z_i=k| \vec w,\vec z_{\neg i})  &  \propto p(z_i=k, w_i =t| \vec w_{\neg i},\vec z_{\neg i}) \\ & = \int p(z_i=k, w_i =t, \vec \theta_d , \vec \beta_k| \vec w_{\neg i},\vec z_{\neg i}) d\vec \theta_d d\vec \beta_k  \\ & =  \int p(z_i=k,  \vec \theta_d |  \vec w_{\neg i},\vec z_{\neg i})p(w_i=t,  \vec \beta_k |  \vec w_{\neg i},\vec z_{\neg i}) d\vec \theta_d d\vec \beta_k  \\ & =  \int p(z_i=k|\vec \theta_d )p( \vec \theta_d |  \vec w_{\neg i},\vec z_{\neg i})p(w_i=t|\vec \beta_k)p(\vec \beta_k |  \vec w_{\neg i},\vec z_{\neg i}) d\vec \theta_d d\vec \beta_k  \\ & = \int p(z_i=k|\vec \theta_d ) Dirichlet(\vec \theta_d | \vec n_{d, \neg i} + \vec \alpha) d\vec \theta_d \\ & * \int p(w_i=t|\vec \beta_k) Dirichlet(\vec \beta_k | \vec n_{k, \neg i} + \vec \eta) d\vec \beta_k \\ & = \int  \theta_{dk} Dirichlet(\vec \theta_d | \vec n_{d, \neg i} + \vec \alpha) d\vec \theta_d  \int \beta_{kt} Dirichlet(\vec \beta_k | \vec n_{k, \neg i} + \vec \eta) d\vec \beta_k \\ & = E_{Dirichlet(\theta_d)}(\theta_{dk})E_{Dirichlet(\beta_k)}(\beta_{kt})\end{split}\end{align}
$$

在上一篇LDA基础里我们讲到了Dirichlet分布的期望公式，因此我们有：

$$
\begin{align}\begin{split}
E_{Dirichlet(\theta_d)}(\theta_{dk}) = \frac{n_{d, \neg i}^{k} + \alpha_k}{\sum\limits_{s=1}^Kn_{d, \neg i}^{s} + \alpha_s}\\
E_{Dirichlet(\beta_k)}(\beta_{kt})= \frac{n_{k, \neg i}^{t} + \eta_t}{\sum\limits_{f=1}^Vn_{k, \neg i}^{f} + \eta_f}
\end{split}\end{align}
$$
最终我们得到每个词对应主题的Gibbs采样的条件概率公式为：
$$
p(z_i=k| \vec w,\vec z_{\neg i})  = \frac{n_{d, \neg i}^{k} + \alpha_k}{\sum\limits_{s=1}^Kn_{d, \neg i}^{s} + \alpha_s}   \frac{n_{k, \neg i}^{t} + \eta_t}{\sum\limits_{f=1}^Vn_{k, \neg i}^{f} + \eta_f}
$$
有了这个公式，我们就可以用Gibbs采样去采样所有词的主题，当Gibbs采样收敛后，即得到所有词的采样主题。
利用所有采样得到的词和主题的对应关系，我们就可以得到每个文档词主题的分布$\theta_d$和每个主题中所有词的分布$\beta_k$。

##### LDA Gibbs采样算法流程总结

现在我们总结下LDA Gibbs采样算法流程。首先是训练流程：

　　　　1） 选择合适的主题数K, 选择合适的超参数向量$\vec \alpha,\vec \eta$ 

　　　　2） 对应语料库中每一篇文档的每一个词，随机的赋予一个主题编号z

　　　　3)  重新扫描语料库，对于每一个词，利用Gibbs采样公式更新它的topic编号，并更新语料库中该词的编号。

　　　　4） 重复第3步的基于坐标轴轮换的Gibbs采样，直到Gibbs采样收敛。

　　　　5） 统计语料库中的各个文档各个词的主题，得到文档主题分布$\theta d$，统计语料库中各个主题词的分布，得到LDA的主题与词的分布$\beta_k$ 。

下面我们再来看看当新文档出现时，如何统计该文档的主题。此时我们的模型已定，也就是LDA的各个主题的词分布βkβk已经确定，我们需要得到的是该文档的主题分布。因此在Gibbs采样时，我们的$E_{Dirichlet(\beta_k)}(\beta_{kt})$已经固定，只需要对前半部分$E_{Dirichlet(\theta_d)}(\theta_{dk})$进行采样计算即可。

　　　　现在我们总结下LDA Gibbs采样算法的预测流程：

　　　　1） 对应当前文档的每一个词，随机的赋予一个主题编号zz

　　　　2)  重新扫描当前文档，对于每一个词，利用Gibbs采样公式更新它的topic编号。

　　　　3） 重复第2步的基于坐标轴轮换的Gibbs采样，直到Gibbs采样收敛。

　　　　4） 统计文档中各个词的主题，得到该文档主题分布。

##### LDA Gibbs采样算法小结 

使用Gibbs采样算法训练LDA模型，我们需要先确定三个超参数$K, \vec \alpha,\vec \eta$。其中选择一个合适的K尤其关键,这个值一般和我们解决问题的目的有关。如果只是简单的语义区分，则较小的KK即可，如果是复杂的语义区分，则K需要较大，而且还需要足够的语料。由于Gibbs采样可以很容易的并行化，因此也可以很方便的使用大数据平台来分布式的训练海量文档的LDA模型。以上就是LDA Gibbs采样算法

##### Tips

懂 LDA 的面试官通常会询问求职者，LDA 中主题数目如何确定？

在 LDA 中，主题的数目没有一个固定的最优解。模型训练时，需要事先设置主题数，训练人员需要根据训练出来的结果，手动调参，有优化主题数目，进而优化文本分类结果。





<font color="red">频率派视角下只有单个骰子，假设每个词的概率，生成一篇语料的概率服从多项分布，直接极大似然估计生成每个词的概率</font>