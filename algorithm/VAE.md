##### 生成模型的目标

<img src="/Users/ftmu/Documents/study/deep_learning/pictures/models/vae/1.png" width=300px>

那现在假设$Z$服从标准的正态分布，那么我就可以从中采样得到若干个$Z_1, Z_2, \dots, Z_n$,然后对它做变换得到$\hat{X}_1 = g(Z_1),\hat{X}_2 = g(Z_2),\dots,\hat{X}_n = g(Z_n)$.判断这个通过$g$构造出来的数据集，它的分布跟我们目标的数据集分布的差异.

##### 经典回顾

首先我们有一批数据样本${X1,…,Xn}$，其整体用$X$来描述，我们本想根据${X1,…,Xn}$得到$X$的分布$p(X)$，如果能得到的话，那我直接根据$p(X)$来采样，就可以得到所有可能的$X$了，这是一个终极理想的生成模型了。当然，这个理想很难实现，于是我们将分布改一改

$$p(X)=\sum_Z p(X|Z)p(Z)\tag{1}$$

<img src="/Users/ftmu/Documents/study/deep_learning/pictures/models/vae/2.png" width=300px>

==问题==: 如果像这个图的话，我们其实完全不清楚：究竟经过重新采样出来的$Zk$，是不是还对应着原来的$Xk$.

##### VAE初现

具体来说，给定一个真实样本$Xk$，我们假设存在**一个专属于$Xk$的分布$p(Z|Xk)$**（学名叫后验分布），并进一步假设这个分布是（独立的、多元的）正态分布。为什么要强调“专属”呢？因为我们后面要训练一个生成器$X=g(Z)$，希望能够把从分布$p(Z|Xk)$采样出来的一个$Zk$原为$Xk$。**现在$p(Z|Xk)$专属于$Xk$，我们有理由说从这个分布采样出来的$Z$应该要还原到$Xk$中去。**

<img src="/Users/ftmu/Documents/study/deep_learning/pictures/models/vae/3.png" width=300px>

==用神经网络拟合后验分布的均值和方差==:
$$
\mu_k = f_1(X_k),\log \sigma_k^2 = f_2(X_k)
$$
有了均值和方差,就能成专属后验分布中采样$Zk$,并经过生成器得到$\hat{X}_k=g(Z_k)$.进而最小化误差$\mathcal{D}(\hat{X}_k,X_k)^2$.

##### 分布标准化

其实**VAE还让所有的$p(Z|X)$都向标准正态分布看齐**，这样就防止了噪声为零，同时保证了模型具有生成能力。怎么理解“保证了生成能力”呢？如果所有的$p(Z|X)$都很接近标准正态分布$N(0,I)$，那么根据定义
$$
p(Z)=\sum_X p(Z|X)p(X)=\sum_X \mathcal{N}(0,I)p(X)=\mathcal{N}(0,I) \sum_X p(X) = \mathcal{N}(0,I)\tag{2}
$$
这样我们就能达到我们的先验假设：$p(Z)$是标准正态分布。然后我们就可以放心地从$N(0,I)$中采样来生成图像了。

<img src="/Users/ftmu/Documents/study/deep_learning/pictures/models/vae/4.png" width=300px>

原论文直接算了一般（各分量独立的）正态分布与标准正态分布的$KL$散度$KL(N(μ,σ2)||N(0,I))$作为这个额外的loss，计算结果为
$$
\mathcal{L}_{\mu,\sigma^2}=\frac{1}{2} \sum_{i=1}^d \Big(\mu_{(i)}^2 + \sigma_{(i)}^2 - \log \sigma_{(i)}^2 - 1\Big)\tag{4}
$$

##### 重参数技巧

我们要从$p(Z|Xk)$中采样一个$Zk$出来，尽管我们知道了$p(Z|Xk)$是正态分布，但是均值方差都是靠模型算出来的，我们要靠这个过程反过来优化均值方差的模型，但是“采样”这个操作是不可导的.
$$
\begin{aligned}&\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right)dz \\ 
=& \frac{1}{\sqrt{2\pi}}\exp\left[-\frac{1}{2}\left(\frac{z-\mu}{\sigma}\right)^2\right]d\left(\frac{z-\mu}{\sigma}\right)\end{aligned}\tag{6}
$$


从$N(μ,σ2)$中采样一个$Z$，相当于从$N(0,1)$中采样一个$ε$，然后让$Z=μ+ε×σ$.

这样一来，“采样”这个操作就不用参与梯度下降了，改为采样的结果参与，使得整个模型可训练了。

##### 代码

```python
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# 算p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)  # 以x为条件的后验概率
z_log_var = Dense(latent_dim)(h)  # 以x为条件的后验概率

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声, 接受均值和方差为输入,变换后,映射到x的维度
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
```

