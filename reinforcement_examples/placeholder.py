import tensorflow as tf

in1 = tf.placeholder(tf.float32)
in2 = tf.placeholder(tf.float32)
output = tf.multiply(in1, in2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={in1: [7.], in2: [2.]}))
