'''import numpy as np
import tensorflow as tf
tf.compat.v1.logging.get_verbosity 
tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)

# 原无上面部分，报错

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../data/",one_hot = True)
print("Training data size: ",mnist.train.num_examples)
print("Validating data size: ",mnist.validation.num_examples)
print("Testing data size: ",mnist.test.num_examples)
print("Example training data: ",mnist.train.images[0])
print("Example training data label: ",mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
# 一个训练batch中的训练个数。数字越小，训练过程越接近随机梯度下降：数字越大越接近梯度下降

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

# 辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。
def inference(input_tensor, avg_class, weights1,biases1,weights2,biases2):
    # 没提供滑动平均类时，直接使用参数当期的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2) + biases2
    else:
        # 首先使用avg_class.average来计算得出变量的滑动平均值，然后计算前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1)) + avg_class.average(biases1)
        )
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class.average(biases2)

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')
    # 我特么受不了了，日，上面第二行的OUTPUT_NODE打成了INPUT_NODE，找了半个小时没找到，日

    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1)
    )
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1)
    )
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE])) 

    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义存储训练轮数的变量，这个变量不需要计算滑动平均值，所以这里指定这个变量为不可训练变量
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    # tf.nn.sparse_softmax_cross_entropy_with_logits用来计算交叉熵
    # 第一个参数是不包括softmax层的前向传播结果，第二个是训练数据的正确答案
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数以及更新每个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))

def main(argv = None):
    mnist = input_data.read_data_sets("../data/",one_hot = True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()





