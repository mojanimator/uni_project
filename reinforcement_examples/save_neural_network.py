import tensorflow as tf

# # save to file
# # need same shape and dtype for restore
W = tf.Variable([[1, 2, 3], [1, 2, 3]], dtype=tf.float32, name="weights")
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name="biases")

init = tf.global_variables_initializer()

# save neural network (weights and biases)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "my_net/save_net.ckpt")
    print("network saved to:", save_path)

# restore neural network (variables)
# redefine same shape same dtype

W1 = tf.Variable(tf.zeros([2, 3]), dtype=tf.float32, name="weights")
b1 = tf.Variable(tf.zeros([1, 3]), dtype=tf.float32, name="biases")

# not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "my_net/save_net.ckpt")
    print('weights', sess.run(W1))
    print('biases', sess.run(b1))
