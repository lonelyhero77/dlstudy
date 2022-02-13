import numpy as np
import tensorflow as tf

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Logistic Regression
# W = tf.Variable(tf.random_normal([2, 1]), name='weight')
# b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# # Nerual Network
# W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
# b1 = tf.Variable(tf.random_normal([2]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
# b2 = tf.Variable(tf.random_normal([1]), name='bias2')
# hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

#NN with tensorboard
with tf.name_scope("layer1") as scope:
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
    
    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b1_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)



# # Wide Nerual Network
# W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
# b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# W2 = tf.Variable(tf.random_normal([10, 1]), name='weight2')
# b2 = tf.Variable(tf.random_normal([1]), name='bias2')
# hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# # Deep Nerual Network
# W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
# b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

# W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
# b2 = tf.Variable(tf.random_normal([10]), name='bias2')
# layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# W3 = tf.Variable(tf.random_normal([10, 1]), name='weight3')
# b3 = tf.Variable(tf.random_normal([1]), name='bias3')
# layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

# W4 = tf.Variable(tf.random_normal([1, 10]), name='weight4')
# b4 = tf.Variable(tf.random_normal([10]), name='bias4')
# layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# W5 = tf.Variable(tf.random_normal([10, 1]), name='weight5')
# b5 = tf.Variable(tf.random_normal([1]), name='bias5')
# hypothesis = tf.sigmoid(tf.matmul(layer4, W5) + b5)

with tf.name_scope("Cost"):    
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_hist = tf.summary.scalar("cost", cost)

with tf.name_scope("train"):
    train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
tf.summary.scalar("accuracy", accuracy)

#tensorboard summarize
summary = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for step in range(5001):
        s, _ = sess.run([summary ,train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(s, global_step=step)
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

        h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
        print("\nHypothesis: {} \nCorrect: {} \nAccuracy: {}".format(h, c, a))


