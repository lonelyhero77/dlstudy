import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import random

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# W = tf.Variable(tf.random_normal([784, nb_classes]))
# b = tf.Variable(tf.random_normal([nb_classes]))
# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Neural Network

W1 = tf.Variable(tf.random_normal([784, 1000]))
b1 = tf.Variable(tf.random_normal([1000]))
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
layer1 = tf.nn.softmax(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([1000, 1000]))
b2 = tf.Variable(tf.random_normal([1000]))
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
# layer2 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([1000, nb_classes]))
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.nn.softmax(tf.matmul(layer2, W3) + b3)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))


training_epochs = 300
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / total_batch

        print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))

    # every tensor has eval() method
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction:", sess.run(
        tf.argmax(hypothesis, 1),
        feed_dict={X: mnist.test.images[r:r + 1]})
    )

    plt.imshow(
        # 784 to 28x28
        mnist.test.images[r:r+1].reshape(28, 28), 
        cmap="Greys", 
        interpolation='nearest'
    )

    plt.show()
