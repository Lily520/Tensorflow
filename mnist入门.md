## mnist入门教程

### 1. 目的
用softmax回归实现mnist数据集手写数字识别

### 2.code

代码文件为 mnist_softmax.py  

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/zhoulili/code/", one_hot=True)
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #布尔值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #布尔值转换成浮点数，求平均
print("******************************************")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

```
### 3.遇到的问题
#### 3.1 出现如下报错信息：
   raise OSError('Not a gzipped file (%r)' % magic)
    OSError: Not a gzipped file (b'<!')
   
   <p style="color:red">错误原因：</p>
 
 第二句话 mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 未在指定路径找到下载的mnist数据集  
 
 <p style="color:red">解决办法：</p>  
 下载该文件，，然后将路径改成数据集所在的路径

### 4.知识点
tf.argmax():返回tensor对象在某一维上的数据最大值所在的索引值  
tf.equal(a,b):返回一组布尔值，若a,b对应位置数值相等，相应位置的结果为True,反之为False  
tf.reduce_mean(): 求平均  

 
 
