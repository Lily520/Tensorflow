import tensorflow as tf
import math
#mnist数据集
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10


def inference(images,hidden1_units,hidden2_units):
    """

    :param images: 输入数据
    :param hidden1_units: 第一个隐含层节点个数
    :param hidden2_units: 第二个隐含层节点个数
    :return: 构建图表，返回包含了预测结果（output prediction）的Tensor
    """

    #hidden1
    with tf.name_scope("hidden1"):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden1_units],
                                                  stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),name="weights")
        bias = tf.Variable(tf.zeros([hidden1_units]),name="bias")
        hidden1 = tf.nn.relu(tf.matmul(images,weights) + bias)

    #hidden2
    with tf.name_scope("hidden2"):
        weights = tf.Variable(tf.truncated_normal([hidden1_units,hidden2_units],
                                                  stddev=1.0/math.sqrt(float(hidden1_units))),name="weights")
        bias = tf.zeros([hidden2_units],name="bias")
        hidden2  = tf.nn.relu(tf.matmul(hidden1,weights) + bias)

    #linear
    with tf.name_scope("linear"):
        weights = tf.Variable(tf.truncated_normal([hidden2_units,NUM_CLASSES],
                                                  stddev=1.0/math.sqrt(float(hidden2_units))),name="weights")
        bias = tf.zeros([NUM_CLASSES],name="bias")
        logits = tf.matmul(hidden2,weights) +bias

    return logits

def loss(logits,labels):
    """

    :param logits: 预测值,float,[batch_size,NUM_CLASSES]
    :param labels: 实际标签值,int32,[batch_size]
    :return: 平均损失
    """

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name="entropy")
    return tf.reduce_mean(cross_entropy,name="entropy_mean")

def training(loss,learning_rate):
    """

    :param loss: 损失值
    :param learning_rate: 用于梯度下降时的学习率
    :return: train_op
    """

    tf.summary.scalar("loss",loss) #记录损失值
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0,name="global_step",trainable=False)
    train_op = optimizer.minimize(loss=loss,global_step=global_step)
    return train_op

def evaluation(logits,labels):
    """

    :param logits: 预测值，float,[batch_size,NUM_CLASSES]
    :param labels: 实际标签值,int32,[batch_size]
    :return: 预测正确的样本个数
    """

    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))