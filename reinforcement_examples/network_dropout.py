import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def add_layer(layer_name, inputs, in_size, out_size, activation_function=None):
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name="W")
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# define placeholder
with tf.name_scope("inputs"):
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, [None, 64], name='x_inputs')  # 8*8
    ys = tf.placeholder(tf.float32, [None, 10], name='y_inputs')

# add output layer
l1 = add_layer("output_layer", xs, 64, 50, activation_function=tf.nn.tanh)
prediction = add_layer("prediction_layer", l1, 50, 10, activation_function=tf.nn.softmax)

# loss between  prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

# summary writer
train_writer = tf.summary.FileWriter("../logs/train", sess.graph)
test_writer = tf.summary.FileWriter("../logs/test", sess.graph)

sess.run(tf.global_variables_initializer())
# train
for i in range(500):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
    if i % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
