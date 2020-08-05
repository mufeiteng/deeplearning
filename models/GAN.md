### GAN

#### 背景

GAN [Goodfellow Ian，GAN] 启发自博弈论中的**二人零和博弈（two-player game）**，由[Goodfellow et al, NIPS 2014](https://arxiv.org/pdf/1406.2661.pdf)开创性地提出。在二人零和博弈中，两位博弈方的利益之和为零或一个常数，即一方有所得，另一方必有所失。GAN模型中的两位博弈方分别由*生成式模型（Generative model）*和*判别式模型（Discriminative model）*充当。

- G是一个生成图片的网络，它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。
- D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

生成模型 G 捕捉样本数据的分布，用服从某一分布（均匀分布，高斯分布等）的噪声 z 生成一个类似真实训练数据的样本，追求效果是越像真实样本越好；判别模型 D 是一个二分类器，估计一个样本来自于训练数据（而非生成数据）的概率，如果样本来自于真实的训练数据，D 输出大概率，否则，D 输出小概率。

在训练过程中，**生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来**。这样，G和D构成了一个动态的“博弈过程”。

最后博弈的结果是什么？**在最理想的状态下**，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

这样我们的目的就达成了：我们得到了一个生成式的模型G，它可以用来生成图片。

![image-20180914220621107](/Users/aszzy/Documents/study/note/pictures/models/gan/example.png)

​	黑色虚线为真实样本的分布，绿色实线为生成样本的分布，蓝色虚线为判别器的概率判断。GAN不断地学习，使得生成器生成的假样本分布越来越接近真实样本，理所当然地，判别器对真假样本的区分也越来越模糊。最终，假样本分布与真实样本一致，而判别器对任何真假样本的判别概率也等于0.5。

#### 数学描述

##### 符号定义

* G (Generator) : 生成器，生成样本混淆D的区分。
* z : 噪声，其先验分布为*p~z~(z)* , （*高斯分布*或者*均匀分布*）
* *p~g~* : 输入为 *z*， G根据 *z* 生成样本 *x* 的概率分布
* *G(z)* : 表示输入为 *z* 下，G生成的样本
* $\theta_g$ : G的参数 
* D (Discriminator) : 辨别器，区分样本来自真实数据还是生成的数据， 参数为 $\theta_d$
* D(x) : 输入来自真实样本 *p~data~* 的概率
* D(G(z)) : 输入来自生成的样本 *p~g~* 的概率

​	生成模型G用于捕捉数据分布，判别模型D用于估计一个样本来自与真实数据而非生成样本的概率。为了学习在真实数据集x上的生成分布$p_g$，生成模型G构建一个从先验分布$p_z(z)$到数据空间的映射函数$G(z,\theta_g)$。 判别模型D的输入是真实图像或者生成图像，$D(x, \theta_d)$输出一个标量，表示输入样本来自训练样本（而非生成样本）的概率。 

##### 优化目标

$$
\min_{G} \max_{D}V(D, G)=\mathbb{E}_{x~\sim  p_{data}(x)}[logD(x)]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z)))].
$$



- 整个式子由两项构成。x表示真实图片，z表示输入G网络的噪声，而G(z)表示G网络生成的图片。
- D(x)表示**D网络判断真实图片是否真实的概率**（因为x就是真实的，所以对于D来说，这个值越接近1越好）。而D(G(z))是D网络判断G生成的图片的是否真实的概率。
- G的目的：上面提到过，D(G(z))是**D网络判断G生成的图片是否真实的概率**，G应该希望自己生成的图片“越接近真实越好”。也就是说，G希望D(G(z))尽可能得大，这时V(D, G)会变小。因此我们看到式子的最前面的记号是min_G。
- D的目的：D的能力越强，D(x)应该越大，D(G(x))应该越小。这时V(D,G)会变大。因此式子对于D来说是求最大(max_D)



* 真正训练时，使用的是 :

$$
\max_{G}\mathbb{E}_{z\sim p_{z}(z)}[logD(G(z))]
$$
因为其梯度性质更好，我们知道，函数$log(x),x\in(0, 1)$在x接近1时的梯度要比接近0时的梯度小很多，接近“饱和”区间。这样，当判别网络D以很高的概率认为生成网络G产生的样本是“假”样本，即$(1-D(G(z, \theta),\phi))\rightarrow 1$。这时目标函数关于$\theta$的梯度反而很小，从而不利于优化。 

* D和G交替训练，固定一方，调整另一方，最终G能估测出真实样本的数据分布，*p~g~* 收敛到 *p~data~*



#### 理论证明

##### 全局最优

$p_{g}=p_{data}$ 

* 固定G，**最优辨别器D**为：

$$
D^*_G(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}
$$

​        固定G，D的训练标准是最大化$V(G, D)$ 
$$
\begin{equation}\begin{split}V(G, D)&=\int_{x}p_{data}(x)log(D(x))dx+\int_{z}p_{z}(z)log(1-D(g(z)))dz\\&=\int_{x}p_{data}(x)log(D(x))+p_{g}(x)log(1-D(x))dx\end{split}\end{equation}
$$
​        函数$y\rightarrow a log(y)+b log(1-y)$在$\frac{a}{a+b}$达到最大值。

* **最优生成器G**：

  注意到D的训练目标是最大化估计条件概率$P(Y=y|x)$的指数似然，y=1代表样本来自$p_{data}$，y=0代表样本来自$p_{g}$

  这时，公式(1)写为：
  $$
  \begin{equation}\begin{split}C(G)&=\max_{D}V(G, D)\\&=\mathbb{E}_{x\sim p_{data}}[logD^*_G(x)]+\mathbb{E}_{z\sim p_z}[log(1-D^*_G(G(z)))]\\&=\mathbb{E}_{x\sim p_{data}}[logD^*_G(x)]+\mathbb{E}_{x\sim p_g}[log(1-D^*_G(x))]\\&=\mathbb{E}_{x\sim p_{data}}[log\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}]+\mathbb{E}_{x\sim p_g}[log\frac{p_g(x)}{p_{data}(x)+g_g(x)}]\\&=\int_{x}p_{data}(x)log(\frac{p_{data}(x)}{p_{data}(x)+f_g(x)})+p_g(x)log(\frac{p_g(x)}{p_{data}(x)+p_g(x)})\\&=-log(4)+KL(p_{data}||\frac{p_{data}+p_g}{2})+KL(p_g||\frac{p_{data}+p_g}{2})\end{split}\end{equation}
  $$
  KL散度大于0，当且仅当$P_{data}=p_g$时达到最小值，$-log(4)$

##### 收敛性

收敛性的证明主要是证明$p_g$的优化过程能够达到全局最优解$p_{data}$ ，将$V(G, D)$改写为关于$p_g$的凸函数$U(p_g, D)$，通过说明凸函数的次导数包含了导数，说明凸函数的优化过程最终能收敛到上面定理1已经证明的唯一全局最优解$p_{data}$上

#### 训练过程

![image-20180914221326456](/Users/aszzy/Documents/study/note/pictures/models/gan/algorithm.png)

* 第一步我们训练D，D是希望V(G, D)越大越好，所以是加上梯度(ascending)

* 第二步训练G时，V(G, D)越小越好，所以是减去梯度(descending)

* 整个训练过程交替进行。

#### 实验部分

实验在MNIST手写数字数据集和多伦多人脸数据集，还有CIFAR-10数据集上对GAN的性能造行了检验

#### 优缺点

##### 优点：

- 根据实际的结果，它们看上去可以比其它模型产生了更好的样本（图像更锐利、清晰）。
- 生成对抗式网络框架能训练任何一种生成器网络（理论上-实践中，用 REINFORCE 来训练带有离散输出的生成网络非常困难）。大部分其他的框架需要该生成器网络有一些特定的函数形式，比如输出层是高斯的。重要的是所有其他的框架需要生成器网络遍布非零质量（non-zero mass）。生成对抗式网络能学习可以仅在与数据接近的细流形（thin manifold）上生成点。
- 不需要设计遵循任何种类的因式分解的模型，任何生成器网络和任何鉴别器都会有用。
- 无需利用马尔科夫链反复采样，无需在学习过程中进行推断（Inference），回避了近似计算棘手的概率的难题。
- 与PixelRNN相比，生成一个样本的运行时间更小。GAN 每次能产生一个样本，而 PixelRNN 需要一次产生一个像素来生成样本。与VAE 相比，它没有变化的下限。如果鉴别器网络能完美适合，那么这个生成器网络会完美地恢复训练分布。换句话说，各种对抗式生成网络会渐进一致（asymptotically consistent），而 VAE 有一定偏置。  与深度玻尔兹曼机相比，既没有一个变化的下限，也没有棘手的分区函数。它的样本可以一次性生成，而不是通过反复应用马尔可夫链运算器（Markov chain operator）。  与 GSN 相比，它的样本可以一次生成，而不是通过反复应用马尔可夫链运算器。  与NICE 和 Real NVE 相比，在 latent code 的大小上没有限制。

##### 缺点：

- **解决不收敛（non-convergence）的问题。** 
  目前面临的基本问题是：所有的理论都认为 GAN 应该在纳什均衡（Nash equilibrium）上有卓越的表现，但梯度下降只有在凸函数的情况下才能保证实现纳什均衡。当博弈双方都由神经网络表示时，在没有实际达到均衡的情况下，让它们永远保持对自己策略的调整是可能的。
- **难以训练：崩溃问题（collapse problem）** 
  GAN模型被定义为极小极大问题，没有损失函数，在训练过程中很难区分是否正在取得进展。GAN的学习过程可能发生崩溃问题（collapse problem），生成器开始退化，总是生成同样的样本点，无法继续学习。当生成模型崩溃时，判别模型也会对相似的样本点指向相似的方向，训练无法继续。
- **无需预先建模，模型过于自由不可控。** 
  与其他生成式模型相比，GAN这种竞争的方式不再要求一个假设的数据分布，即不需要formulate p(x)，而是使用一种分布直接进行采样sampling，从而真正达到理论上可以完全逼近真实数据，这也是GAN最大的优势。然而，这种不需要预先建模的方法缺点是太过自由了，对于较大的图片，较多的 pixel的情形，基于简单 GAN 的方式就不太可控了。在GAN中，每次学习参数的更新过程，被设为D更新k回，G才更新1回，也是出于类似的考虑。

 #### 读后思考

* GAN里的**G**和**D**看上去互相依存，我的输出输入你，你的输出又输入给我，这显然是一个**先有鸡还是先有蛋**的问题，其实在Algorithm 1. 里面已经写的很清楚了，在算法的一开始，将非常粗糙的人工噪音数据z和真实数据x两次输入鉴别函数**D**，在最优化过程中对它们的组合计算值(算法1给出的公式里有)做最大化，这样k步之后（大循环内部的小循环），再开始优化**G**（其实没有**D**，**G**根本无法接收BP结果来优化自己，也说明了这个问题）。那么，如果没有人工噪音数据z，怎么解决冷启动问题？直接随机化一个z好了
* 引言中还有一句很经典的话，我意译一下：BP算法和dropout机制带来了神经网络和深度学习近些年的辉煌。细想一下，确实如此

#### Tensorflow 实现

1. D将真实图片编码为特征向量，全连接后得到打分，label为1，得到交叉熵d_real
2. 随机采样噪声向量z，G转置卷积，将z变成图片,为Gz。经过D编码，得到向量，进行打分，label为0，得到交叉熵d_fake
3. G希望Gz尽可能接近真实图片，故打分为D(G(z))，label为1，交叉熵为g_loss
4. 先训练D k次，在训练G，交替进行

##### Discriminator

输入：图片

输出：向量

##### Generator

输入：向量

输出：图片

##### Loss

```python
z_dimensions = 100
batch_size = 50

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator
Gz = generator(batch_size, z_dimensions)
# Gz holds the generated images
Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images
Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images
# Define losses
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx))
)
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg))
)
g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg))
)

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)
# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

```

##### Train

```python
# Pre-train discriminator
for i in range(300):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: real_image_batch})

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

    # Train discriminator on both real and fake images
    _, __ = sess.run([d_trainer_real, d_trainer_fake],
                                           {x_placeholder: real_image_batch})

    # Train generator
    _ = sess.run(g_trainer)

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        summary = sess.run(merged, {x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
```



### CGAN

#### 背景

​        [原始GAN](#jump)提出，与其他生成式模型相比，GAN这种竞争的方式不再要求一个假设的数据分布，即不需要formulate p(x)，而是使用一种分布直接进行采样sampling，从而真正达到理论上可以完全逼近真实数据，这也是GAN最大的优势。然而，这种不需要预先建模的方法缺点是太过自由了，对于较大的图片，较多的 pixel的情形，基于简单 GAN 的方式就不太可控了。

​	为了解决GAN太过自由这个问题，一个很自然的想法是给GAN加一些约束，于是便有了[Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)。这项工作提出了一种带条件约束的GAN，在生成模型（D）和判别模型（G）的建模中均引入**条件变量y**（conditional variable y），使用额外信息y对模型增加条件，可以指导数据生成过程。这些条件变量y可以基于多种信息，例如类别标签，用于图像修复的部分数据，来自不同模态（modality）的数据，这样使得GAN能够更好地被应用于跨模态问题，例如图像自动标注。如果条件变量y是类别标签，可以看做CGAN 是把纯无监督的 GAN 变成有监督的模型的一种改进。这个简单直接的改进被证明非常有效,并广泛用于后续的相关工作中。[Mehdi Mirza et al.](https://arxiv.org/abs/1411.1784) 的工作是在MNIST数据集上以类别标签为条件变量，生成指定类别的图像。

#### 本文思想

​	把噪声z和条件y作为输入同时送进G，生成跨域向量，再通过非线性函数映射到数据空间。把数据x和条件y作为输入同时送进D，生成跨域向量，并进一步判断x是真实训练数据的概率。

​	优化函数为：
$$
\min_G\max_DV(D, G)=\mathbb{E}_{x\sim p_{data}(x)}[logD(x|y)]+\mathbb{E}_{z\sim p_{z}(z)}[log(1-D(G(z|y)))]
$$

#### 实验

##### MNIST数据集实验

​	在MNIST上以数字类别标签为约束条件，最终根据类别标签信息，生成对应的数字。

​	生成模型的输入是100维服从均匀分布的噪声向量，条件y是类别标签的one hot编码。噪声z和标签y分别映射到隐层（200和1000个unit），在被第二次映射前，连接了所有120个unit。最终用一个sigmoid层输出784维(28*28)的单通道图像。

​	判别模型把输入图像x映射到一个有240个unit和5 pieces的maxout layer，把y映射到有50个unit和5pieces的maxout layer。同时，把所有隐层连在一起成为sigmoid层之前的有240个unit和4pieces的maxout layer。最终的输出是该样本x来自训练集的概率。

##### Flickr数据集上的图像自动标注实验(Mir Flickr-25k)

​	在ImageNet数据集(21000labels)上预训练模型，采用最后一层全连接输出作为图像特征。文本表示是选取YFCC数据集，预处理后用skip-model训练成200维的词向量，除去出现次数少于200的词，最后剩下247465个词。

​	实验是基于MIR Flickr数据集，利用上面的模型提取图像和文本特征。为了便于评价，对于每个图片我们生成了100的标签样本，对于每个生成标签利用余弦相似度找到20个最接近的词，最后是选取了其中10个最常见的词。

​	在实验中，效果最好的生成器是接收100维的高斯噪声把它映射到500维的ReLu层，同时把4096维的图像特征向量映射到一个2000维的ReLu隐层，再上面的两种表示连接在一起映射到一个200维的线性层，最终由这个层输出200维的仿标签文本向量。

​	判别器是由500维和1200维的ReLu隐层组成，用于处理文本和图像。最大输出层是有1000个单元和3spieces的连接层用于给最终的sigmoid层处理输入数据。

#### 总结

现在大多数网络数据都存在标签缺失的情况，例如这篇论文用到的MIR Flickr数据集，25000张图片的标签中，出现20次以上的词才1300多，而且其中有很多标签词并不与图片内容有关，例如notmycat，图片中奖杯信息，蛋糕上人名等词语。所以文章中提到的能够自动生成标签的方法很有意义。但是，分析最后生成标签的结果，还有有不少生成标签是与图像内容无关的，考虑可以想办法改进。

#### Tensorflow实现

##### Generator

```python
# G(z)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        cat1 = tf.concat([x, y], 1)
        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)
        dense2 = tf.layers.dense(relu1, 784, kernel_initializer=w_init)
        o = tf.nn.tanh(dense2)
        return o
```

##### Discriminator

```python
# D(x)
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        cat1 = tf.concat([x, y], 1)
        dense1 = tf.layers.dense(cat1, 128, kernel_initializer=w_init)
        lrelu1 = lrelu(dense1, 0.2)
        dense2 = tf.layers.dense(lrelu1, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense2)
        return o, dense2
```

##### Loss

```python
# variables : input
x = tf.placeholder(tf.float32, shape=(None, 784))
y = tf.placeholder(tf.float32, shape=(None, 10))
z = tf.placeholder(tf.float32, shape=(None, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))
```

### DCGAN

#### 架构

[论文地址](https://arxiv.org/pdf/1511.06434.pdf) DCGAN的原理和GAN是一样的，这里就不在赘述。它只是把上述的G和D换成了两个卷积神经网络（CNN）。但不是直接换就可以了，DCGAN对卷积神经网络的结构做了一些改变，以提高样本的质量和收敛的速度，这些改变有：

- 取消所有pooling层。G网络中使用转置卷积（transposed convolutional layer）进行上采样，D网络中用加入stride的卷积代替pooling。
- 在D和G中均使用batch normalization
- 去掉FC层，使网络变为全卷积网络
- G网络中使用ReLU作为激活函数，最后一层使用tanh
- D网络中使用LeakyReLU作为激活函数

**DCGAN中的G网络示意：**

![image-20180914222111667](/Users/aszzy/Documents/study/note/pictures/models/gan/dcgan.png)

#### Tensorflow实现

##### Generator

```python
# G(z)
def generator(x, y_label, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_label], 3)

        # 1st hidden layer
        deconv1 = tf.layers.conv2d_transpose(
            cat1, 256, [7, 7], strides=(1, 1), padding='valid', 
            kernel_initializer=w_init, bias_initializer=b_init
        )
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        # 2nd hidden layer
        deconv2 = tf.layers.conv2d_transpose(
            lrelu1, 128, [5, 5], strides=(2, 2), padding='same', 
            kernel_initializer=w_init, bias_initializer=b_init
        )
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)

        # output layer
        deconv3 = tf.layers.conv2d_transpose(
            lrelu2, 1, [5, 5], strides=(2, 2), padding='same', 
            kernel_initializer=w_init, bias_initializer=b_init
        )
        o = tf.nn.tanh(deconv3)
        return o
```

##### Discriminator

```python
# D(x)
def discriminator(x, y_fill, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer
        cat1 = tf.concat([x, y_fill], 3)

        # 1st hidden layer
        conv1 = tf.layers.conv2d(
            cat1, 128, [5, 5], strides=(2, 2), padding='same', 
            kernel_initializer=w_init, bias_initializer=b_init
        )
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(
            lrelu1, 256, [5, 5], strides=(2, 2), padding='same', 
            kernel_initializer=w_init, bias_initializer=b_init
        )
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # output layer
        conv3 = tf.layers.conv2d(
            lrelu2, 1, [7, 7], strides=(1, 1), padding='valid', kernel_initializer=w_init
        )
        o = tf.nn.sigmoid(conv3)

        return o, conv3
```

##### Loss

```python
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 10))
y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 10))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, y_label, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain)
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)

# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

```




