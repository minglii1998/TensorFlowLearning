import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf 

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 在训练时神经网络会创建这些变量，测试时会通过保存的模型加载这些变量的取值
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights",shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
    )

    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))

    # 因为这里原本写在if下了，所以在test的时候weights一直都是None，就一直出错，醉了
    return weights

def inference(input_tensor, regularizer):

    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE,LAYER1_NODE],regularizer
        )
        biases = tf.get_variable(
            "biases",[LAYER1_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        

    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE,OUTPUT_NODE],regularizer
        )
        biases = tf.get_variable(
            "biases",[OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0)
        )
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2