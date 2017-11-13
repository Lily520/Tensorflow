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