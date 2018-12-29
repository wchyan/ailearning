import tensorflow as tf
import mnist_dataset
import mnist_plt

#replace the mnist data path to yours
mnist_data_path="/Users/yanweichuan/workspace/dev/tensorflow/data/mnist"
mnist_test_data = mnist_dataset.test(mnist_data_path)
mnist_train_data = mnist_dataset.train(mnist_data_path)

batch_size = 100
learn_rate = 0.03

# define input/output and W, b
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.truncated_normal([784, 10], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
b = tf.Variable(tf.truncated_normal([10], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)

# define model, cost and optimizer
y_ = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # train
    for epoch in range(5):
        mnist_data_iter = mnist_train_data.batch(batch_size).make_one_shot_iterator()
        batch_xs, batch_ys = mnist_data_iter.get_next()
        # Fit training using batch data
        acc = []
        try:
            for _ in range(1000):
                xs, ys = sess.run([batch_xs, batch_ys])
                ys = sess.run(tf.one_hot(ys, 10))
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                _, c, acc_val  = sess.run([optimizer, cost, accuracy], feed_dict={x: xs.reshape([-1, 784]), y: ys})
                acc.append(acc_val)
        except tf.errors.OutOfRangeError as e:
            pass
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.4f}".format(c), "accuracy=%f" % sess.run(tf.reduce_mean(acc)))

    # test
    mnist_data_iter = mnist_test_data.batch(30).make_one_shot_iterator()
    batch_xs, batch_ys = mnist_data_iter.get_next()
    xs, ys = sess.run([batch_xs, batch_ys])
    pv = sess.run(tf.argmax(y_, 1), feed_dict={x: xs.reshape([-1, 784])})
    ys = sess.run(tf.one_hot(ys, 10))
    mnist_plt.plot_image_label_prediction(xs.reshape([-1, 784]), ys, pv)
