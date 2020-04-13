---
layout: post
title:  "Deep Learning Framework Programming Model"
---

## Before Deep Learning Framework

在深度学习框架（Deep Learning Framework, DLF）普及前，实验室的研究员基本上用的是 MATLAB 和 Python 来做实验。在训练一个模型前，研究员首先要手推导数，再把公式转化成对应的代码。比如研究员需要训练一个拥有两层 fully connected layer 的神经网络来做 MNIST 的 Auto Encoder，他需要书写如下 MATLAB 代码

```matlab
% Forward -------------------------------------------------
numImages = size(data,2);
numImages_inv = 1./numImages;

features = sigmoid(W1*data+repmat(b1,[1,numImages]));
output = W2*features+repmat(b2,[1,numImages]);
mean_act1 = mean(features,2);

squared_error = 0.5.*(output - data).^2;
cost = numImages_inv .* sum(squared_error(:));
cost = cost + 0.5 .* lambda .* (sum(W1(:).^2) + sum(W2(:).^2));
cost = cost + beta .* sum(kl_div(sparsityParam, mean_act1));

% Backward -------------------------------------------------
delta2 = -(data-output);
b2grad = mean(delta2,2);
W2grad = numImages_inv .* delta2 * features' + lambda .* W2;

delta_sparsity = repmat(beta.*(-sparsityParam./mean_act1+(1-sparsityParam)./(1-mean_act1)),[1,numImages]);
delta1 = (W2' * delta2 + delta_sparsity) .* features .* (1-features);
b1grad = mean(delta1,2);
W1grad = numImages_inv .* delta1 * data' + lambda .* W1;

% Update -------------------------------------------------
...
```

我们可以看到，整个代码还是很复杂的。对简单的网络这么做还可以接受，但网络复杂了之后比较难处理了。尤其是当模型训练不好的时候，你不知道是导数求错了，还是模型没有设计好，或是优化的hyper parameter没有调好。十分难Debug。

但当时没有一个好的深度学习框架，大家也都是这么做的。直到Caffe的出现。

## 2013: 有点像黑箱的Caffe

Caffe 实现了[自动求导(autodiff)](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)。用户只需要用plaintext/protobuf定义[前向的网络](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick.prototxt)和[训练参数](https://github.com/BVLC/caffe/blob/master/examples/cifar10/cifar10_quick_solver.prototxt)，然后运行`caffe train filename.prototxt`就可以开始训练了，整个过程用户一行脚本代码都不用写。这种 programming model 一般被称为 [define and run](https://docs.chainer.org/en/stable/guides/define_by_run.html)，意为我们先定义一个模型，然后在去执行这个模型。

Caffe 提供的自动求导大大简化了 deep learning 的研究过程，是一款相当受欢迎的深度学习框架。但同时，Caffe 也为 autodiff 付出了不小的代价。

首先是 Caffe 的训练过程很黑箱。在 MATLAB 中，用户可以完全控制整个训练过程：在任意地方设置断点，然后更改或是可视化某一个 tensor。但在 Caffe 中，数据只能通过RecordIO读取。其次是 Caffe 没有控制流（control flow），较难表示 RNN 这类模型结构会因输入而变化的模型。

而 TensorFlow 的出现，很好滴解决了 Caffe 的这些限制。

## 2015: TensorFlow

TensorFlow也实现了自动求导。

和 Caffe 类似，TensorFlow 也是 [define and run](https://docs.chainer.org/en/stable/guides/define_by_run.html) 的 programming model。用户在host language（Python/R/C++）把计算图（包括 Forward 和 Optimization）定义好，然后在每个 iteration 通过`session.run()`来 feed 和 fetch 计算图里的 Tensor。TensorFlow 支持更细的定义力度，用户可以直接定义 operators（如加减乘除，concat，convolution）。

```python
import numpy as np
import tensorflow as tf

N, D = 3, 4

with tf.device('/gpu:0'):
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)
	z = tf.placeholder(tf.float32)

	a = x * y
	b = a + z
	c = tf.reduce_sum(b)

grad_x, grad_y, grad_z = tf.gradients(c, [x,y,z])

with tf.Session() as sess:
	values = {
		x: np.random.randn(N, D),
		y: np.random.randn(N, D),
		z: np.random.randn(N, D),
	}
	out = sess.run([c, grad_x, grad_y, grad_z],
                   feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out
```

与 Caffe 的黑箱不同的是，TensorFlow 在构造计算图的时候有更丰富的信息。比如，用户可以在构造计算图的时候查看各个 Tensor 的 dimension 并做检查。另外，TensorFlow 提供的 [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) 能很好地可视化模型。

TensorFlow 支持了 control flow（`cond`/`while_loop`）。但由于 TensorFlow 的计算图独立于 host language，这些 control flow 写起来和 host language 的语法有很大差别。

```python
i = tf.constant(0)
c = lambda i: tf.less(i, 10)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])
```

## 2017: PyTorch

PyTorch 同样实现了自动求导。

但与 Caffe 和 TensorFlow 不同，PyTorch 是 [define by run](https://docs.chainer.org/en/stable/guides/define_by_run.html) 的 programming model。PyTorch 会根据 Python **代码的执行**生成计算图，而不是先定义计算图再执行，所以代码会简练很多。很多熟悉 Python 的用户会发现 PyTorch 相比 TensorFlow 容易上手。

```python
import torch
from torch.autograd import Variable

N, D = 3, 4
x = Variable(torch.randn(N, D).cuda(), requires_grad = True)
y = Variable(torch.randn(N, D).cuda(), requires_grad = True)
z = Variable(torch.randn(N, D).cuda(), requires_grad = True)

a = x * y
b = a + z
c = torch.sum(b)

c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
```

由于是逐行执行 Python 的代码，整个 forward 过程十分透明，用户可以用熟悉的 Python Debugger 进行调试。PyTorch 使用的是 Python 的 control flow，使用起来十分的自然。比如for loop能很直接实现一个RNN

```python
outputs = []
hiddens = []
for t in range(time_steps):
  x_input = X[t]
  output, hidden = your_cell(X_input)
  outputs.append(output)
  hidden.append(hidden)
```

对于自动生成的 backward，用户也可以通过 [register\_hook](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.register_hook) 来调试，大大增加了易用性，可以说是基本上解决了黑箱问题。

## Conclusion

从 Programming Model 方面来看，DLF 主要的贡献是 autodiff。在另一方面，由于 DL 又是一个很吃计算力的领域，所以 DLF 都会一套接口支持 CPU/GPU 计算。
