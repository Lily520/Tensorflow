## 深入mnist

### 1.目的
用多层卷积神经网络实现mnist数据集手写数字识别。   
此代码实现的网络结构为： 卷积+池化(14x14x32)+卷积+池化(7x7x64)+全连接+Dropout+softmax

### 2.code
```python
import tensorflow as tf

#加载minist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/zhoulili/tensors/", one_hot=True)

####### 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

###### 卷积和池化
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pooling_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


if __name__ == "__main__":

    #占位符
    x = tf.placeholder("float",shape=[None,784])
    y_ = tf.placeholder("float",shape = [None,10])

    #第一层卷积 输入28x28x1  第一层卷积后的尺寸 14x14x32
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x,[-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pooling_2x2(h_conv1)

    #第二层卷积 卷积后的尺寸为 7x7x64
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pooling_2x2(h_conv2)

    #密集连接层 1024个神经元的全连接层
    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

    #Dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    #输出层
    W_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

    #训练和评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction,"float"))  #测试时用

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0}) #神经元被选中的概率为1.0
            print("step:" , i  , ", train_accuarcy: " , train_accuracy)
        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    #计算测试误差
    good = 0 #预测正确的数目
    total = 0 #总数目
    for i in range(200):
        batch = mnist.test.next_batch(50)
        good += accuracy_sum.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
        total += batch[0].shape[0]
    print("test_accuracy: " , float(good)/float(total))

###结果：test_accuracy:  0.9925
```

### 3.遇到的问题
#### 3.1 ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[10000,28,28,32]
原因：内存不够用   
解决办法：   
将最后一行计算测试误差的代码
```python
print("test_accuracy: " , accuracy.eval(feed_dict = {x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
```
替换成：  
```python
accuracy_sum = tf.reduce_sum(tf.cast(correct_prediction,"float")) #测试时用
good = 0 #预测正确的数目
total = 0 #总数目
for i in range(200):
batch = mnist.test.next_batch(50)
good += accuracy_sum.eval(feed_dict = {x:batch[0],y_:batch[1],keep_prob:1.0})
total += batch[0].shape[0]
print("test_accuracy: " , float(good)/float(total))
```

#### 4.知识点
##### 4.1. tf.truncated_normal(shape,mean,stddev)
这个函数产生一个截断的正态分布，就是说产生正态分布的值如果与均值的差值大于两倍的标准差，那就重新生成。  
参数说明：
>shape:表示生成张量的维度   
>mean:是均值  
>stddev:是标准差   

##### 4.2. 卷积函数 tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=None,name=None)
返回一个Tensor,即我么你常说的feature map    
参数说明：   
>name:指定该操作的name   
>input:需要做卷积的输入图像，shape=[batch,in_height,in_width,in_channels]   
>>batch：训练时一个batch的图片数量 
>>in_height: 图片高度  in_width:图片宽度  in_channels:图像通道数  

>filter:CNN中的卷积核，shape=[filter_height,filter_width,in_channels,out_channels]  
>>shape四个维度分别代表：卷积核高度，卷积核宽度，图像通道数，卷积核个数  

>strides:卷积时在图像每一维的步长，一维向量，长度为4 [1,strides,strides,1]  
>padding:SAME/VALID  
>>VALID方式，feature map 大小：  
>>>out_height=(in_height-filter_height+1)/strides[1]  
>>>out_width=(in_width-filter_width+1)/strides[2]  

>use_cudnn_on_gpu:是否使用gpu加速，默认为true  

##### 4.3. tf.nn.max_pool(value, ksize, strides, padding, name=None) 
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式   
参数是四个，和卷积很类似：  
>第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape  
>第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1  
>第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]  
>第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'  
 

##### 4.4. tf.reduce_sum(input_tensor,reduction_indices=None,keep_dims=False,name=None)  
计算输入tensor元素的和，或者按照reduction_indices指定的轴进行求和  

假设 x=[[1,1,1]  
       [1,1,1]]  
tf.reduce_sum(x): 6  
tf.reduce_sum(x,0) : [2,2,2] (缩减0,意味着将行压扁成一条)  
tf.reduce_sum(x,1) : [3,3]  
tf.reduce_sum(x,1,keep_dims=True) :[[3],[3]]  
tf.reduce_sum(x,[0,1]) : 6  
