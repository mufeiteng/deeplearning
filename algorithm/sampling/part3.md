### Gibbs Sampling

#### 重新寻找合适的细致平稳条件

从二维数据分布开始，假设$\pi(x_1,x_2)$是一个二维联合概率分布， 观察第一个特征维度相同的两个点$A(x_1^{(1)},x_2^{(1)})$和$A(x_1^{(1)},x_2^{(2)})$ ，容易发现下面两式成立：
$$
\begin{align}
\begin{split}
\pi(x_1^{(1)},x_2^{(1)})\pi(x_2^{(2)}|x_1^{(1)})&=\pi(x_1^{(1)})\pi(x_2^{(1)}|x_1^{(1)})\pi(x_2^{(2)}|x_1^{(1)})\\
\pi(x_1^{(1)},x_2^{(2)})\pi(x_2^{(1)}|x_1^{(1)})&=\pi(x_1^{(1)})\pi(x_2^{(2)}|x_1^{(1)})\pi(x_2^{(1)}|x_1^{(1)})
\end{split}
\end{align}
$$
所以有：
$$
\pi(x_1^{(1)},x_2^{(1)})\pi(x_2^{(2)}|x_1^{(1)})=\pi(x_1^{(1)},x_2^{(2)})\pi(x_2^{(1)}|x_1^{(1)})
$$
即：
$$
\pi(A)\pi(x_2^{(2)}|x_1^{(1)})=\pi(B)\pi(x_2^{(1)}|x_1^{(1)})
$$
观察上式再观察细致平稳条件的公式，我们发现在$x_1=x_1^{(1)}$这条直线上如果用条件概率分布$\pi(x_2|x_1^{(1)})$作为马尔可夫链的状态转移概率，则任意两个点之间的转移满足细致平稳条件 ！同样的道理，在$x_2=x_2^{(1)}$这条直线上，如果用条件概率分布$\pi(x_1|x_2^{(1)})$ 作为马尔可夫链的状态转移概率，则任意两个点之间的转移也满足细致平稳条件 。那是因为假如有一点$C(x_1^{(2)},x_2^{(1)})$，我们可以得到：
$$
\pi(A)\pi(x_1^{(2)}|x_2^{(1)})=\pi(C)\pi(x_1^{(1)}|x_2^{(1)})
$$
基于上面的发现，我们可以这样构造分布$\pi(x_1,x_2)$的马尔可夫链对应的状态转移矩阵$P$ ：
$$
\begin{align}
\begin{split}
P(A \to B)&=\pi(x_2^{(B)}|x_1^{(1)}) \quad if \space x_1^{(A)}=x_1^{(B)}=x_1^{(1)}\\
P(A \to C)&=\pi(x_1^{(C)}|x_2^{(1)}) \quad if \space x_2^{(A)}=x_2^{(C)}=x_2^{(1)}\\
P(A\to D)&=0 \quad else
\end{split}
\end{align}
$$
有了这个状态转移矩阵，我们很容易验证平面上的任意两点$E,F$，满足细致平稳条件：
$$
\pi(E)P(E\to F)=\pi(F)P(F\to E)
$$

#### 二维Gibbs采样

利用上一节找到的状态转移矩阵，我们就得到了二维Gibbs采样，这个采样需要两个维度之间的条件概率。具体过程如下：

1: 输入平稳分布$\pi(x_1,x_2)$，设定状态转移次数阈值$n_1$ ，需要的样本个数$n_2$ 

2: 随机初始化状态值$x_1^{(1)}$ 和$x_2^{(1)}$

3: $for \space t=0 \to n_1+n_2-1$:

- 从条件概率分布$P(x_2|x_1^{(t)})$中采样得到样本$x_2^{(t+1)}$
- 从条件概率分布$P(x_1|x_2^{(t+1)})$中采样得到样本$x_1^{(t+1)}$

样本集$\{(x_1^{(n_1)},x_2^{(n_1)}),(x_1^{(n_1+1)},x_2^{(n_1+1)}),\cdots,(x_1^{(n_1+n_2-1)},x_2^{(n_1+n_2-1)})\}$ 就是我们需要的平稳分布对应的样本集。

采样是在两个坐标轴上不停的轮换的。当然，坐标轴轮换不是必须的，我们也可以每次随机选择一个坐标轴进行采样。不过常用的Gibbs采样的实现都是基于坐标轴轮换的。 

#### 多维Gibbs采样

上面的这个算法推广到多维的时候也是成立的。比如一个n维的概率分布 $\pi(x_{1},x_{2},...,x_{n})$，可以通过在$n$个坐标轴上轮换采样，来得到新的样本。对于轮换到的任意一个坐标轴 $x_{i}$ 上的转移，马尔科夫链的状态转移概率为 $P(x_{i}|x_{1},x_{2},...,x_{i-1},x_{i+1},...,x_{n})$，即固定$n−1$个坐标轴，在某一个坐标轴上移动。具体的算法过程如下：

1: 输入平稳分布$\pi(x_1,x_2,\cdots,x_n)$，设定状态转移次数阈值$n_1$ ，需要的样本个数$n_2$ 

2: 随机初始化状态值$(x_1^{(1)},x_2^{(1)},\cdots,x_n^{(1)})$  

3: $for \space t=0 \to n_1+n_2-1$:

- 从条件概率分布$P(x_1|x_2^{(t)},x_3^{(t)},\cdots,x_n^{(t)})$中采样得到样本$x_1^{(t+1)}$
- 从条件概率分布$P(x_2|x_1^{(t+1)},x_3^{(t+1)},\cdots,x_n^{(t+1)})$中采样得到样本$x_2^{(t+1)}$
- $\cdots$
- 从条件概率分布$P(x_j|x_1^{(t+1)},x_2^{(t+1)},\cdots,x_{j-1}^{(t+1)},x_{j+1}^{(t+1)},\cdots,x_n^{(t+1)})$中采样得到样本$x_j^{(t+1)}$
- $\cdots$
- 从条件概率分布$P(x_n|x_1^{(t+1)},x_2^{(t+1)},\cdots,x_{n-1}^{(t+1)})$中采样得到样本$x_n^{(t+1)}$

样本集$\{(x_1^{(n_1)},x_2^{(n_1)},\cdots,x_n^{(n_1)}),\cdots,(x_1^{(n_1+n_2-1)},x_2^{(n_1+n_2-1)},\cdots,x_n^{(n_1+n_2-1)})\}$就是我们需要的平稳分布对应的样本集。

整个采样过程和$Lasso$回归的坐标轴下降法算法非常类似，只不过$Lasso$回归是固定$n−1$个特征，对某一个特征求极值。而$Gibbs$采样是固定$n−1$个特征在某一个特征采样。

#### 二维Gibbs采样实例

假设我们要采样的是一个二维正态分布 $Norm(\mu,\Sigma)$，其中：
$$
\begin{align}\begin{split}
\mu&=(\mu_1,\mu_2)=(5,-1)\\
\Sigma&=\left(\begin{array}{c} 
    \sigma^2_1 & \rho\sigma_1\sigma_2\\ 
    \rho\sigma_1\sigma_2 & \sigma^2_2 \\ 
\end{array}\right) =
\left(\begin{array}{c} 
1 & 1\\
1 & 4
\end{array}\right)     
\end{split}\end{align}
$$
而采样过程中的需要的状态转移条件分布为：
$$
\begin{align}
\begin{split}
P(x_1|x_2)=Norm(\mu_1+\rho\sigma_1/\sigma_2(x_2-\mu_2),1-\rho^2\sigma^2_1)\\
P(x_2|x_1)=Norm(\mu_2+\rho\sigma_2/\sigma_1(x_1-\mu_1),1-\rho^2\sigma^2_2)
\end{split}\end{align}
$$

#### Gibbs采样小结

由于Gibbs采样在高维特征时的优势，目前我们通常意义上的MCMC采样都是用的Gibbs采样。

当然Gibbs采样是从M-H采样的基础上的进化而来的，同时Gibbs采样要求数据至少有两个维度，一维概率分布的采样是没法用Gibbs采样的,这时M-H采样仍然成立。

有了Gibbs采样来获取概率分布的样本集，有了蒙特卡罗方法来用样本集模拟求和，他们一起就奠定了MCMC算法在大数据时代高维数据模拟求和时的作用。 

### 采样方法总结

问题：已知随机变量$X$的概率密度为$p(x)$，$f(x)$ 为$X$的函数，$E(f(X))=\int_a^b f(x) p_X(x)dx$ 。 

根据MCMC原理，如果能从$p(x)$中抽样$x_i$，然后对这些$f(x_i)$取平均即可近似$f(x)$的期望 $E_N(f)=\frac{1}{N}\sum_{i=1}^N\frac{f(x_i)}{p(x_i)})$ 。 

#### 逆变换采样

通过CDF与PDF的关系，得到样本$x_i$ 。

**缺点**：

- 有时CDF不好求
- CDF的反函数不好求

#### 接受拒绝采样

从容易抽样的$q(x)$中抽样，以一定的方法拒绝某些样本，达到接近$p(x)$分布的目的。

**最终得到$n$个接受的的样本$x_0,x_1,\cdots,x_n$，则最后的蒙特卡洛求解结果为：$\frac1n\sum^n_{i=1}\frac{f(x_i)}{q(x_i)}$ 。**

**缺点**：

- 合适的$ q $分布比较难以找到
- 很难确定一个合理的 $c$ 值。

#### 重要性采样

从容易抽样的概率分布$q(x)$中抽样，不需要拒绝样本。

**最终得到$n$个接受的的样本$x_0,x_1,\cdots,x_n$，则最后的蒙特卡洛求解结果为：$\frac1n\sum^n_{i=1}\frac{f(x_i)\cdot p(x)}{q(x_i)}$ 。**

#### 基于马尔科夫链采样

对于给定的概率分布$p(x)$,我们希望能有便捷的方式生成它对应的样本。获得样本之后即可对期望进行蒙特卡洛近似。如果我们能构造一个转移矩阵为$P$的马氏链，使得该马氏链的平稳分布恰好是$p(x)$, 那么我们从任何一个初始状态$x_0$出发沿着马氏链转移, 得到一个转移序列$x_0,x_1,...,x_n,x_{n+1},...$， 如果马氏链在第$n$步已经收敛了，于是我们就得到了$p(x)$的样本$x_n,x_{n+1},...$。

**重要的问题：随意给定一个平稳分布$\pi$，即目标分布$p(x)$  ,如何得到它所对应的马尔科夫链状态转移矩阵$P$呢？**

#### MCMC采样

给定的概率分布$p(x)$, 一个转移矩阵为$Q$马氏链，构造新的转移矩阵$Q'$，使得$p(x)$是$Q’$的平稳分布。新的马氏链 $Q’ $ 的转移概率为 $q(i,j)\alpha(i,j)$ ， 其中$\alpha(i,j) = p(j)q(j,i)$。

这时，从$Q’$的边缘转移概率采样，得到的样本服从$p(x)$ 的概率分布。但是**$Q’$的边缘转移概率**含有$p(x)$，不易采样。

采用拒绝采样的做法，目的：从$q(x_t,x)\alpha(x_t,x)$中采样，选择的 proposal distribution为$q(x_t,x)$，因为$\alpha(x_t,x)<1$，故选择常量$c=1$ ，这时，接受概率为：
$$
\alpha=\frac{q(x_t,x)\alpha(x_t,x)}{c\cdot q(x_t,x)}=\alpha(x_t,x)
$$
故从均匀分布$(0, 1)$中抽样得到$u$，如果$u<\alpha(x_t,y)=p(y)q(y\vert x_t)$，则接受转移$x_t\rightarrow y$ 。

 **缺点**：由于$\alpha(x_t,y)$ 可能非常的小，比如0.1，导致我们大部分的采样值都被拒绝转移，采样效率很低。

#### M-H采样

经过对上述 MCMC 采样算法中接受率的微小改造，令$α(i,j)=min(\frac{p(j)Q(j,i)}{p(i)Q(i,j)},1)$。

**缺点**：

- 数据特征非常多，需要计算接受率，在高维时计算量大。并且由于接受率的原因导致算法收敛时间变长；
- 由于特征维度大，特征的条件概率分布好求，但是特征的联合分布不好求。

#### Gibbs采样

采用了轮换坐标轴的方法进行采样，能够处理高维特征。要求数据至少有两个维度，一维概率分布的采样是没法用Gibbs采样的,这时M-H采样仍然成立。

有了Gibbs采样来获取概率分布的样本集，有了蒙特卡罗方法来用样本集模拟求和，他们一起就奠定了MCMC算法在大数据时代高维数据模拟求和时的作用。