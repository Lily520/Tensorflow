import mnist
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import argparse
import sys
FLAGS = None

def placeholder_inputs(batch_size):
    """

    :param batch_size: 一次处理的样本个数
    :return: images_palceholder,labels_placeholder
    """
    images_placeholder = tf.placeholder(tf.float32,shape=(batch_size,mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batch_size))
    return images_placeholder,labels_placeholder

def fill_feed_dict(dataSet,images_pl,labels_pl):
    """

    :param dataSet: mnist数据集
    :param images_pl: images_placeholder
    :param labels_pl: labels_placeholder
    :return: feed_dict
    """
    images_feed,labels_feed = dataSet.next_batch(FLAGS.batch_size,FLAGS.fake_data)
    feed_dict = {
        images_pl:images_feed,
        labels_pl:labels_feed
    }
    return feed_dict

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,dataSet):
    """
    在数据集上进行一次评估
    :param sess:
    :param eval_correct: 预测正确的样本个数
    :param images_placeholder:
    :param labels_placeholder:
    :param dataSet:

    """
    true_count = 0 #测试正确的样本个数
    num_epoch = dataSet.num_examples // FLAGS.batch_size #循环次数
    num_examples = num_epoch * FLAGS.batch_size #参与评估的样本个数
    for i in range(num_epoch):
        feed_dict = fill_feed_dict(dataSet,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct,feed_dict=feed_dict)
    precision = float(true_count)/float(num_examples)
    print("num_examples:",num_examples,", num_correct:",true_count,", precision:",precision)

def run_training():

    dataSet = input_data.read_data_sets(FLAGS.input_data_dir,FLAGS.fake_data)
    with tf.Graph().as_default():
        images_placeholder,labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = mnist.inference(images_placeholder,FLAGS.hidden1,FLAGS.hidden2)
        loss = mnist.loss(logits,labels_placeholder)
        train_op = mnist.training(loss,FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits,labels_placeholder)

        summary = tf.summary.merge_all() #汇总所有summary.scalar的节点
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir,sess.graph) #汇总数据写入磁盘
        sess.run(init)

        #训练
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(dataSet.train,images_placeholder,labels_placeholder)

            for key in feed_dict.keys():
                print(key,":",feed_dict[key].shape)

            _,loss_value = sess.run([train_op,loss],feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 100 == 0:
                print("step:",step,"  loss:",loss_value,"  duration:",duration)
                summary_str = sess.run(summary,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str,step) #没训练100次，进行一次合并汇总
                summary_writer.flush()

            if (step+1) % 1000 == 0 or (step+1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir,"model.ckpt")
                saver.save(sess,checkpoint_file,global_step=step)

                #evaluate training set
                print("training data eval:")
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,dataSet.train)
                #evaluate validation set
                print("validation data eval:")
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,dataSet.validation)
                #evaluate test set
                print("test data eval:")
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, dataSet.test)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate",type=float,default=0.01,help="Initial learning rate")
    parser.add_argument("--max_steps", type=int, default=2000, help="Number of steps to run training")
    parser.add_argument("--hidden1", type=int, default=128, help="Number of units in hidden layer 1")
    parser.add_argument("--hidden2", type=int, default=32, help="Number of units in hidden layer 2")
    parser.add_argument("--batch_size", type=int, default=100, help="NBatch size.  Must divide evenly into the dataset sizes.")
    parser.add_argument("--input_data_dir", type=str, default=os.path.join(os.getenv('/home/zhoulili', '/tmp'),
                           'tensorflow/mnist/input_data'), help="Directory to put the input data.")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getenv('/home/zhoulili', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'), help="Directory to put the log data.")
    parser.add_argument("--fake_data", default=False, help="If true, uses fake data for unit testing.",action = "store_true")


    FLAGS,unparsed = parser.parse_known_args()
    print("FLAGS.log_dir:",FLAGS.log_dir)
    print("FLAGS.input_data_dir:",FLAGS.input_data_dir)
    tf.app.run(main=main,argv=[sys.argv[0]] + unparsed)

##结果：
'''
training data eval:
num_examples: 55000 , num_correct: 49266 , precision: 0.8957454545454545
validation data eval:
num_examples: 5000 , num_correct: 4526 , precision: 0.9052
test data eval:
num_examples: 10000 , num_correct: 9018 , precision: 0.9018
'''