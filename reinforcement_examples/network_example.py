import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


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
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs


# make real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # add one column
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data, y_data)
# plt.show()

# define placeholder
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_inputs')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_inputs')

# add hidden layer
l1 = add_layer("hidden", xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
prediction = add_layer("output", l1, 10, 1, activation_function=None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# initialise variables
init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", graph=sess.graph)

sess.run(init)

# plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()  # unblock plotting process
plt.show()
# train
for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)

        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.5)
