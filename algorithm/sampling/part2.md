### 马尔可夫过程

MCMC(Markov Chain Monte Carlo)的基础理论为马尔可夫过程，在MCMC算法中，为了在一个指定的分布上采样，根据马尔可夫过程，首先从任一状态出发，模拟马尔可夫过程，不断进行状态转移，最终收敛到平稳分布。

**用蒙特卡罗方法随机模拟来求解一些复杂的连续积分或者离散求和的方法，但是这个方法需要得到对应的概率分布的样本集，而想得到这样的样本集很困难。因此我们需要马尔科夫链来帮忙。** [参考链接1](https://cosx.org/2013/01/lda-math-mcmc-and-gibbs-sampling) [参考链接2](http://www.cnblogs.com/pinard/p/6625739.html)

#### 马尔可夫链

设$X_t$表示随机变量$X$在离散时间$t$时刻的取值。若该变量随时间变化的转移概率仅仅依赖于它的当前取值，即
$$
P(X_{t+1}=s_j|X_0=s_0,X_1=s_1,...,X_t=s_t)=P(X_{t+1}=s_j|X_t=s_t)
$$
也就是说状态转移的概率只依赖于前一个状态，称这个变量为马尔可夫变量。这个性质称为马尔可夫性质，具有马尔可夫性质的随机过程称为马尔可夫过程。

马尔可夫链指的是在一段时间内随机变量$X$的取值序列$(X_0,X_1,...,X_m)$，它们满足如上的马尔可夫性质。

既然某一时刻状态转移的概率只依赖于它的前一个状态，那么我们只要能求出系统中任意两个状态之间的转换概率，这个马尔科夫链的模型就定了。

#### 状态转移概率

这个马尔科夫链是表示股市模型的，共有三种状态：牛市(Bull market), 熊市(Bear market)和横盘(Stagnant market)。

<img src="/Users/ftmu/Documents/study/DL/pictures/algorithm/markov_chain.png" height="200px">

每一个状态都以一定的概率转化到下一个状态。比如，牛市以0.025的概率转化到横盘的状态。这个状态概率转化图可以以矩阵的形式表示。如果我们定义矩阵 $P$某一位置 $P(i,j)$的值为 $P(j|i)$，即从状态$i$转化到状态$j$的概率，并定义牛市为状态0， 熊市为状态1, 横盘为状态2. 这样我们得到了马尔科夫链模型的状态转移矩阵为：
$$
P={\begin{bmatrix}0.9&0.075&0.025\\0.15&0.8&0.05\\0.25&0.25&0.5\end{bmatrix}}
$$

#### 状态转移矩阵的性质

假设我们当前股市的概率分布为：$[0.3,0.4,0.3]$ ,即30%概率的牛市，40%概率的熊盘与30%的横盘。然后这个状态作为序列概率分布的初始状态$t_0$,将其带入这个状态转移矩阵计算 $t_{1},t_{2},...$ 的状态。代码如下：

```python
import numpy as np
matrix = np.matrix([[0.9, 0.075, 0.025], [0.15, 0.8, 0.05], [0.25, 0.25, 0.5]], dtype=float)
vector = np.matrix([0.3, 0.3, 0.4], dtype=float)
for i in range(100):
    vector = vector*matrix
    print('current round: ', vector)
```

可以发现，从第60轮开始，我们的状态概率分布就不变了，一直保持在$[0.625, 0.3125, 0.0625]$，即62.5%的牛市，31.25%的熊市与6.25%的横盘。初始化概率分布$[0.7, 0.1, 0.2]$，结果保持不变。

同时，对于一个确定的状态转移矩阵$P$，它的$n$次幂$P^n$在当$n$大于一定的值的时候也可以发现是确定的，

也就是说我们的马尔科夫链模型的状态转移矩阵收敛到的稳定概率分布与我们的初始状态概率分布无关。如果我们得到了这个稳定概率分布对应的马尔科夫链模型的状态转移矩阵，则我们可以用任意的概率分布样本开始，带入马尔科夫链模型的状态转移矩阵，这样经过一些序列的转换，最终就可以得到符合对应稳定概率分布的样本。这个性质不光对我们上面的状态转移矩阵有效，对于绝大多数的其他的马尔科夫链模型的状态转移矩阵也有效。同时不光是离散状态，连续状态时也成立。

#### 马尔科夫链收敛定理

**马氏链定理：**如果一个非周期马氏链具有转移概率矩阵 $P$, 且它的任何两个状态是连通的，那么$\lim_{n\rightarrow \infin}p^n_{ij}$存在且与$i$无关，记$\lim_{n\rightarrow \infin}p^n_{ij}=\pi$， 我们有

1）
$$
\lim_{n\rightarrow \infin}P_{ij}^n=\pi(j)
$$
2)
$$
\lim_{n\rightarrow \infin}P^n=\begin{bmatrix} \pi(1) & \cdots & \pi(2) & \cdots & \pi(j) & \cdots \\ \pi(1) & \cdots & \pi(2) & \cdots & \pi(j) & \cdots \\ \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \\ \pi(1) & \cdots & \pi(2) & \cdots & \pi(j) & \cdots \\ \cdots & \cdots & \cdots & \cdots & \cdots & \cdots \end{bmatrix}
$$
3)
$$
\pi^{t+1}_{(j)}=\sum^{\infin}_{i=0}\pi^t_{(i)}P_{ij}
$$
4) $\pi$是方程$\pi P=\pi$的唯一非负解，其中： 
$$
\pi = [\pi(1), \pi(2), \cdots, \pi(j),\cdots ], \quad \sum_{i=0}^{\infty} \pi_i = 1
$$
上面的性质中需要解释的有：

- 非周期的马尔科夫链：这个主要是指马尔科夫链的状态转化不是循环的，如果是循环的则永远不会收敛。幸运的是我们遇到的马尔科夫链一般都是非周期性的。用数学方式表述则是：对于任意某一状态$i$, $d$为集合 $\left\{ n|n\geq 1,P_{ii}^{n}>0 \right\}$ 的最大公约数，如果$d=1$，则该状态为非周期的。
- 任何两个状态是连通的：是指存在一个 $n$，使得矩阵$P^n$中的任何一个元素的数值都大于零。
- 马尔科夫链的状态数可以是有限的，也可以是无限的。因此可以用于连续概率分布和离散概率分布。
- $\pi$通常称为马尔科夫链的平稳分布。
- 我们用$X_i$表示在马氏链上跳转第$i$步后所处的状态，如果$ \lim_{n\rightarrow\infin}P^n_{ij}=\pi (j)$存在，很容易证明以上定理的第3个结论。由于 

$$
\begin{align}
\begin{split}
P(X_{n+1}=j) & = \sum_{i=0}^\infty P(X_n=i) P(X_{n+1}=j|X_n=i) \\
& = \sum_{i=0}^\infty P(X_n=i) P_{ij}
\end{split}
\end{align}
$$

上式两边取极限就得到$\displaystyle \pi^{t+1}(j) = \sum_{i=0}^{\infty}\pi^t(i)P_{ij}$ 。假设状态的数目为$n$，则有$\pi^{t+1}=\pi^t\cdot P$，其中 $\pi^{t}=(\pi^{(t)}_1,\pi^{(t)}_2,\cdots,\pi^{(t)}_n）$。

- 稳定分布与特征向量的关系:

稳定分布$\pi$是一个(行)向量，它的元素都非负且和为1，不随施加$P$操作而改变，定义为 $\pi P=\pi$。

那么：$P^T\pi^T=\pi^T$ ，

对比定义可以看出,这两个概念是相关的,并且$\pi=\frac{e}{\sum_i e_i}$是由($\sum_i\pi_i=1$)归一化的转移矩阵$P$的左特征向量$e$的倍数，其特征值为1.

操作上：

1. 对$P$的转置进行特征值分解得到特征向量和特征值.
2. 最大的特征值应为1,其对应的特征向量为矩阵的第$i$列
3. 对特征向量进行归一化,可以得到该状态转移矩阵的稳定分布

演示代码如下:

```python
w,v=np.linalg.eig(P.transpose())
idx = np.where(abs(w-1.0)<1e-9)[0][0]
print w[idx]
print v[:,idx]/sum(v[:,idx])
```

实际上$n*n$矩阵特征向量的一种求法就是用一个随机向量不断迭代与该矩阵相乘。

#### 基于马尔科夫链采样

对于给定的概率分布$p(x)$,我们希望能有便捷的方式生成它对应的样本。由于马氏链能收敛到平稳分布， 于是一个很的漂亮想法是：如果我们能构造一个转移矩阵为$P$的马氏链，使得该马氏链的平稳分布恰好是$p(x)$, 那么我们从任何一个初始状态$x_0$出发沿着马氏链转移, 得到一个转移序列$x_0,x_1,...,x_n,x_{n+1},...$， 如果马氏链在第$n$步已经收敛了，于是我们就得到了$p(x)$的样本$x_n,x_{n+1},...$。

从初始概率分布 $\pi^0$ 出发，我们在马氏链上做状态转移，记$X_i$的概率分布为$\pi^i$，则有
$$
\begin{align}
\begin{split}
X_0 & \sim \pi^0(x)\\
X_i & \sim \pi^i(x)\\
\quad\quad \pi^i(x) &= \pi^{i-1}(x)P = \pi^0(x)P^n
\end{split}
\end{align}
$$
由马氏链收敛的定理, 概率分布$\pi^i(x)$将收敛到平稳分布$\pi_x$。假设到第$n$步的时候马氏链收敛，则有
$$
\begin{align}
\begin{split}
X_0 & \sim \pi^0(x) \\
X_1 & \sim \pi^1(x) \\
& \cdots \\
X_n & \sim \pi^n(x)=\pi(x) \\
X_{n+1} & \sim \pi(x) \\
X_{n+2}& \sim \pi(x) \\
& \cdots
\end{split}
\end{align}
$$
所以 $X_n,X_{n+1},X_{n+2},\cdots \sim \pi(x)$都是同分布的随机变量，当然他们并不独立。如果我们从一个具体的初始状态 $x_0$ 开始, 沿着马氏链按照概率转移矩阵做跳转，那么我们得到一个转移序列 $x_0, x_1, x_2, \cdots x_n, x_{n+1}\cdots$ 由于马氏链的收敛行为，$x_n, x_{n+1},\cdots$ 都将是平稳分布 $\pi(x)$ 的样本。

**总结下基于马尔科夫链的采样过程：**

- 输入Markov Chain状态转移概率矩阵$P$，设定状态转移次数阈值$n_1$ ，需要的样本个数$n_2$ ；
- 从任意简单概率分布采样得到初始状态值$x_0$ ；
- $for \space 0 \to \space n_1+n_2-1: $ 从概率分布$P(x|x_t)$中采样得到样本$x_{t+1}$ ；

样本集$(x_{n_1},\cdots,x_{n_1+n_2-1})$即为$P$平稳分布$\pi$对应的样本集；

**如果假定我们可以得到我们需要采样样本的平稳分布所对应的马尔科夫链状态转移矩阵，那么我们就可以用马尔科夫链采样得到我们需要的样本集，进而进行蒙特卡罗模拟。但是一个重要的问题是，随意给定一个平稳分布$\pi$，即目标分布$p(x)$  ,如何得到它所对应的马尔科夫链状态转移矩阵$P$呢？**

### MCMC采样

在马尔科夫链中我们讲到给定一个概率平稳分布$\pi$ , 很难直接找到对应的马尔科夫链状态转移矩阵$P$。而只要解决这个问题，我们就可以找到一种通用的概率分布采样方法，进而用于蒙特卡罗模拟。

马氏链的收敛性质主要由转移矩阵$P$决定, 所以**基于马氏链做采样的关键问题是如何构造转移矩阵$P$, 使得平稳分布恰好是我们要的分布$p(x)$ **。如何能做到这一点呢？我们主要使用如下的定理。

#### 细致平稳条件

如果非周期马氏链的转移矩阵$P$和分布 $\pi(x)$ 满足:
$$
\pi(i)P_{ij} = \pi(j)P_{ji} \quad\quad \text{for all} \quad i,j
$$
则$\pi(x)$是马氏链的平稳分布，上式被称为细致平稳条件 。

其实这个定理是显而易见的，因为细致平稳条件的物理含义就是对于任何两个状态$i,j$，从$i$转移出去到$j$ 而丢失的概率质量，恰好会被从$j$转移回$i$的概率质量补充回来，所以状态 $i$ 上的概率质量$\pi(i)$是稳定的，从而 $\pi(x)$是马氏链的平稳分布。数学上的证明也很简单，由细致平稳条件可得:
$$
\begin{align}\begin{split} & \sum_{i=1}^\infty \pi(i)P_{ij} = \sum_{i=1}^\infty \pi(j)P_{ji} = \pi(j) \sum_{i=1}^\infty P_{ji} = \pi(j) \\ & \Rightarrow \pi P = \pi \end{split}\end{align}
$$
由于 $\pi$是方程$\pi P=\pi $ 的解，所以 $\pi$是$P$ 的平稳分布。

**注: 细致平稳条件为马尔可夫链有平稳分布的充分条件**

假设马氏链$Q$为另一个马尔科夫链的转移核，并且是一个容易抽样的分布，被称之为建议分布 ($q(i,j)$ 表示从状态$i$转移到状态 $j$的概率，也可以写为$q(j|i)$或者 $q(i\to j)$). $p(x)$为目标概率分布,我们需要从$p(x)$中采样. 显然，通常情况下
$$
p(i)q(i,j) \neq p(j)q(j,i)
$$
也就是细致平稳条件不成立，所以$p(x)$不太可能是这个马氏链的平稳分布。我们可否对马氏链做一个改造，使得细致平稳条件成立呢？譬如，我们引入一个$\alpha(i,j)$，我们希望:
$$
p(i)q(i,j)\alpha(i,j) = p(j)q(j,i)\alpha(j,i)
$$
取什么样的 $\alpha(i,j)$ 以上等式能成立呢？最简单的，按照对称性，我们可以取:
$$
\begin{align}\begin{split}\alpha(i,j) = p(j)q(j,i)\\\alpha(j,i) = p(i)q(i,j)\end{split}\end{align}
$$
所以有:
$$
p(i)P'(i,j)=p(j)P'(j,i)
$$
其中：
$$
\begin{align}\begin{split}P'(i,j)&=q(i,j)\alpha(i,j)\\P'(j,i)&=q(j,i)\alpha(j,i)\end{split}\end{align}
$$


于是我们把原来具有转移矩阵 $Q$ 的一个很普通的马氏链，改造为了具有转移矩阵$P'$ 的马氏链，而$P’$ 恰好满足细致平稳条件，由此马氏链 $P'$  的平稳分布就是 $p(x)$ ！$P'$就是$p(x)$的转移核.

在改造$Q$ 的过程中引入的$\alpha(i,j)$ 称为接受率，物理意义可以理解为在原来的马氏链上，从状态$i$ 以 $q(i,j)$ 的概率转跳转到状态$j$ 的时候，我们以 $α(i,j)$ 的概率接受这个转移，于是得到新的马氏链 $P’ $ 的转移概率为 $q(i,j)\alpha(i,j)$。

为了使$q(i,j)\alpha(i,j)$满足细致平稳条件. 一般来说$q(i,j)\alpha(i,j)$也是不方便直接采样的. 实际的做法是采用拒绝采样方法,把 $\alpha(i,j)$看作一个状态转移的接受概率. 从$(0,1)$均匀分布中做一个采样得到$u$,如果$u<\alpha(i,j)$则接受$q(i,j)$采样出样本的状态转移,否则拒绝，保持原状态。

**马氏链转移和接受概率:**

假设我们已经有一个转移矩阵$Q$(对应元素为$q(i,j)$ ), 把以上的过程整理一下，我们就得到了如下的用于采样概率分布$p(x)$的算法。

1: 输入任意选定的马尔可夫链状态转移矩阵$Q$，目标分布$\pi (x)$ ，状态转移次数阈值$n_1$ ，需要的样本个数$n_2$ ；

2: 从任意简单概率分布采样得到初始状态值$x_0$；

3: $for \space 0 \to \space n_1+n_2-1: $ 

- 从条件概率分布$Q(x|x_t)$中采样得到样本$x_* $
- 从均匀分布采样$u\sim Uniform[0,1]$
- 如果$u<\alpha(x_t,x_*)=\pi(x_*)Q(x_*,x_t)$，则接受转移$x_t\rightarrow x_* $，即$x_{t+1}=x_*$
- 否则拒绝转移，即$t=max\{t-1,0\}$

上述过程中$\pi(x),Q(i|j)$ 说的都是离散的情形，事实上即便这两个分布是连续的，以上算法仍然是有效，于是就得到更一般的连续概率分布$\pi (x)$ 的采样算法，而$Q(i|j)$ 就是任意一个连续二元概率分布对应的条件分布。

由于 $\alpha(x_t,x_*)$ 可能非常的小，比如0.1，导致我们大部分的采样值都被拒绝转移，采样效率很低。有可能我们采样了上百万次马尔可夫链还没有收敛，也就是上面这个$n_1$要非常非常的大，这让人难以接受，怎么办呢？这时就轮到我们的M-H采样出场了。

### M-H采样

M-H算法主要是解决接受率过低的问题, 回顾MCMC采样的细致平稳条件： 
$$
p(i)q(i,j)\alpha(i,j)=p(j)q(j,i)\alpha(j,i)
$$
我们采样效率低的原因是$\alpha(i,j)$ 太小了，比如$\alpha(i,j)$为0.1，而 $\alpha(j,i)$为0.2. 即： 
$$
p(i)Q(i,j)×0.1=p(j)Q(j,i)×0.2
$$
如果两边同时扩大五倍，接受率提高到了0.5，但是细致平稳条件却仍然是满足的，即：
$$
p(i)Q(i,j)×0.5=p(j)Q(j,i)×1
$$
这样我们的接受率可以做如下改进，即：  
$$
α(i,j)=min(\frac{p(j)Q(j,i)}{p(i)Q(i,j)},1)
$$
于是，经过对上述 MCMC 采样算法中接受率的微小改造，我们就得到了如下教科书中最常见的 Metropolis-Hastings 算法。

1: 任意选定的建议分布（状态转移核）$Q$ ，抽样的目标分布密度函数$\pi(x)$ ，状态转移次数阈值$n_1$ ，需要的样本个数$n_2$ ；

2: 从任意简单概率分布采样得到初始状态值$x_0$ ；

3: $for \space 0 \to \space n_1+n_2-1: $

- 从条件概率分布$Q(x|x_t)$ 中采样$x_*$
- 从均匀分布采样$u\sim Uniform[0,1]$  
- 如果$u<\alpha(x_t,x_*)=\min\{1,\frac{\pi(j)Q(j\vert i)}{\pi(i)Q(i\vert j)}\}$ ，则接受转移$x_t\rightarrow x_*$，即$x_{t+1}x_*=$
- 否则拒绝转移，即$t=max\{t-1,0\}$  

#### 建议分布

很多时候我们选择的转移矩阵Q都是对称的,即$q(i,j)=q(j,i)$，这时接受率可以进一步简化为
$$
\alpha (i,j)=\{1,\frac{\pi(j)q(j,i)}{\pi(i)q(i,j)}\}=\{1,\frac{\pi(j)}{\pi(i)}\}
$$
特别地，$q(i,j)=q(|i−j|) $被称为随机游走 Metropolis 算法，例子
$$
q(j,i)\propto \exp(-\frac{(j-i)^2}{2})
$$
读者也很容易发现，当正态分布的方差为常数，均值为 $i$，参数为 $j$ 的这些转移核都满足这种类型。这种类型的转移核的特点是，当 $j$ 在均值 $i$  附近的时候，其概率也就越高。

**举例：**我们的目标平稳分布是一个均值3，标准差2的正态分布，而选择的马尔可夫链状态转移矩阵$Q(i,j)$的条件转移概率是以$i$为均值,方差1的正态分布在位置$j$的值。

```python
# -*- coding:utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt


def mh(q, p, m, n):
    # randomize a number
    x = random.uniform(0.1, 1)
    for t in range(0, m+n):
        x_sample = q.sample(x)
        try:
            accept_prob = min(1, p.prob(x_sample)*q.prob(x_sample, x)/(p.prob(x)*q.prob(x, x_sample)))
        except:
            accept_prob = 0

        u = random.uniform(0, 1)

        if u < accept_prob:
            x = x_sample

        if t >= m:
            yield x


class Exponential(object):
    def __init__(self, scale):
        self.scale = scale
        self.lam = 1.0 / scale

    def prob(self, x):
        if x <= 0:
            raise Exception("The sample shouldn't be less than zero")

        result = self.lam * np.exp(-x * self.lam)
        return result

    def sample(self, num):
        sample = np.random.exponential(self.scale, num)
        return sample


# 假设我们的目标概率密度函数p1(x)是指数概率密度函数
scale = 5
p1 = Exponential(scale)


class Norm():
    def __init__(self, mean, std):
        self.mean = mean
        self.sigma = std

    def prob(self, x):
        return np.exp(-(x - self.mean) ** 2 / (2 * self.sigma ** 2.0)) * 1.0 / (np.sqrt(2 * np.pi) * self.sigma)

    def sample(self, num):
        sample = np.random.normal(self.mean, self.sigma, size=num)
        return sample

# 假设我们的目标概率密度函数p1(x)是均值方差分别为3,2的正态分布
p2 = Norm(3, 2)


class Transition():
    def __init__(self, sigma):
        self.sigma = sigma

    def sample(self, cur_mean):
        cur_sample = np.random.normal(cur_mean, scale=self.sigma, size=1)[0]
        return cur_sample

    def prob(self, mean, x):
        return np.exp(-(x-mean)**2/(2*self.sigma**2.0)) * 1.0/(np.sqrt(2 * np.pi)*self.sigma)


# 假设我们的转移核方差为10的正态分布
q = Transition(10)

m = 100
n = 100000 # 采样个数

simulate_samples_p1 = [li for li in mh(q, p1, m, n)]

plt.subplot(2,2,1)
plt.hist(simulate_samples_p1, 100)
plt.title("Simulated X ~ Exponential(1/5)")

samples = p1.sample(n)
plt.subplot(2,2,2)
plt.hist(samples, 100)
plt.title("True X ~ Exponential(1/5)")

simulate_samples_p2 = [li for li in mh(q, p2, m, n)]
plt.subplot(2,2,3)
plt.hist(simulate_samples_p2, 50)
plt.title("Simulated X ~ N(3,2)")


samples = p2.sample(n)
plt.subplot(2,2,4)
plt.hist(samples, 50)
plt.title("True X ~ N(3,2)")

plt.suptitle("Transition Kernel N(0,10)simulation results")
plt.show()
```

<img src="/Users/ftmu/Documents/study/DL/pictures/algorithm/mh_eg.jpg" height="300px"></img>

M-H采样完整解决了使用蒙特卡罗方法需要的任意概率分布样本集的问题，因此在实际生产环境得到了广泛的应用。但是在大数据时代，M-H采样面临着两大难题：

- 数据特征非常多，需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长；
- 由于特征维度大，特征的条件概率分布好求，但是特征的联合分布不好求。

这时候我们能不能只有各维度之间条件概率分布的情况下方便的采样呢？

