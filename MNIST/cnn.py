import tensorflow as tf
import mnist_dataset
import mnist_plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


#replace the mnist data path to yours
mnist_data_path="/Users/yanweichuan/workspace/dev/tensorflow/data/mnist"
mnist_test_data = mnist_dataset.test(mnist_data_path)
mnist_train_data = mnist_dataset.train(mnist_data_path)

input_unit = 784
output_unit = 10
learn_rate = 1e-4
batch_size = 100
keep_prop_val = 0.8
epoches = 5

x = tf.placeholder(tf.float32, [None, input_unit])
y = tf.placeholder(tf.float32, [None, output_unit])
keep_prop = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prop)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=1))
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # train
    for epoch in range(epoches):
        mnist_data_iter = mnist_train_data.batch(batch_size).make_one_shot_iterator()
        batch_xs, batch_ys = mnist_data_iter.get_next()
        acc = []
        try:
            for _ in range(1000):
                xs, ys = sess.run([batch_xs, batch_ys])
                ys = sess.run(tf.one_hot(ys, output_unit))
                _, c, acc_val = sess.run([optimizer, cross_entropy, accuracy], feed_dict={x: xs, y: ys, keep_prop: keep_prop_val})
                acc.append(acc_val)
        except tf.errors.OutOfRangeError as e:
            pass
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.4f}".format(c), "accuracy=%f" % sess.run(tf.reduce_mean(acc)))

    # test
    mnist_data_iter = mnist_test_data.batch(batch_size).make_one_shot_iterator()
    batch_xs, batch_ys = mnist_data_iter.get_next()
    xs, ys = sess.run([batch_xs, batch_ys])
    pv = sess.run(tf.argmax(y_conv, 1), feed_dict={x: xs, keep_prop:1.0})
    ys = sess.run(tf.one_hot(ys, output_unit))
    mnist_plt.plot_image_label_prediction(xs.reshape([-1, input_unit]), ys, pv)
