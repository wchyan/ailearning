import tensorflow as tf
import mnist_dataset
import mnist_plt

#replace the mnist data path to yours
mnist_data_path="/Users/yanweichuan/workspace/dev/tensorflow/data/mnist"
mnist_test_data = mnist_dataset.test(mnist_data_path)
mnist_train_data = mnist_dataset.train(mnist_data_path)

input_unit = 784
hidden_unit = 100
output_unit = 10

epoches = 5
batch_size = 100
learn_rate = 0.03
keep_prop_val = 0.8

x = tf.placeholder(tf.float32, [None, input_unit], name="input_x")
y = tf.placeholder(tf.float32, [None, output_unit], name="output_y")
keep_prop = tf.placeholder(tf.float32, name="keep_prop")

W1 = tf.Variable(tf.truncated_normal([input_unit, hidden_unit], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
b1 = tf.Variable(tf.truncated_normal([hidden_unit], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)

W2 = tf.Variable(tf.truncated_normal([hidden_unit, output_unit], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
b2 = tf.Variable(tf.truncated_normal([output_unit], mean=0.0, stddev=0.1, dtype=tf.float32), dtype=tf.float32)

# define model, cost and optimizer
y1_ = tf.nn.relu(tf.matmul(x, W1) + b1)
y1_drop = tf.nn.dropout(y1_, keep_prop)
y_ = tf.nn.softmax(tf.matmul(y1_drop, W2) + b2)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

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
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                _, c, acc_val = sess.run([optimizer, cost, accuracy], feed_dict={x: xs.reshape([-1, input_unit]), y: ys, keep_prop: keep_prop_val})
                acc.append(acc_val)
        except tf.errors.OutOfRangeError as e:
            pass
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.4f}".format(c), "accuracy=%f" % sess.run(tf.reduce_mean(acc)))

    # test
    mnist_data_iter = mnist_test_data.batch(batch_size).make_one_shot_iterator()
    batch_xs, batch_ys = mnist_data_iter.get_next()
    xs, ys = sess.run([batch_xs, batch_ys])
    pv = sess.run(tf.argmax(y_, 1), feed_dict={x: xs.reshape([-1, input_unit]), keep_prop:1.0})
    ys = sess.run(tf.one_hot(ys, output_unit))
    mnist_plt.plot_image_label_prediction(xs.reshape([-1, input_unit]), ys, pv)
