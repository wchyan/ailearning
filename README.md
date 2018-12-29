
# MNIST

在tensorflow学习中，MNIST数据集是入门级的数据集。在各种教材中，都是使用的tensorflow.examples.tutourials.mnist或者其他早期的包。tensoflow的版本更新很快，导致该包在不同的版本上不能运行。在学习过程中，将MNIST的数据处理独立出来，可以更加详细的了解数据的加载处理过程，并加上了Logistic Regression, MLP和CNN的练习例子，源码在github上下载。

* 数据：mnist_dataset.py可以自行下载，也可在 http://yann.lecun.com/exdb/mnist/ 下载，在使用中指定目录即可
* 在Logistic Regression, MLP及 CNN对比中，从精度可以看出CNN的威力，同样训练次数，CNN的精度可以达到99%

运行结果：
$ python logistic_regression.py<br>
Epoch: 0001 cost= 0.6083 accuracy=0.781133<br>
Epoch: 0002 cost= 0.4808 accuracy=0.876433<br>
Epoch: 0003 cost= 0.4314 accuracy=0.888667<br>
Epoch: 0004 cost= 0.4046 accuracy=0.894733<br>
Epoch: 0005 cost= 0.3875 accuracy=0.898317<br>

$ python mlp.py<br>
Epoch: 0001 cost= 0.5855 accuracy=0.746833<br>
Epoch: 0002 cost= 0.4982 accuracy=0.864967<br>
Epoch: 0003 cost= 0.4333 accuracy=0.887717<br>
Epoch: 0004 cost= 0.3097 accuracy=0.901783<br>
Epoch: 0005 cost= 0.3945 accuracy=0.909750<br>

$ python cnn.py<br>
Epoch: 0001 cost= 0.2593 accuracy=0.879617<br>
Epoch: 0002 cost= 0.2437 accuracy=0.968667<br>
Epoch: 0003 cost= 0.2490 accuracy=0.979500<br>
Epoch: 0004 cost= 0.1930 accuracy=0.984783<br>
Epoch: 0005 cost= 0.1982 accuracy=0.988584<br>

