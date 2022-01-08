import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1, 2, 3]
y_data = [1, 2, 3]
computerpower = -3

W = tf.Variable(tf.random_normal([1]), name="weight")
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

#poorman's method
learning_rate = 0.1
gradient = tf.reduce_mean( (W*X-Y)*X )
descent = W - learning_rate * gradient
update = W.assign(descent)

#i love computing power!
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1)
train = optimizer.minimize(cost)

#using computing power in handy way
gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

#creating session and initialize variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Using poorman's method")
for step in range(21):
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    # sess.run(train, feed_dict={X: x_data, Y: y_data})

# Weight initialization
# sess.run(W.assign(tf.random_normal([1])))

sess.run(W.assign([computerpower]))
print("\n Current Value", sess.run(W))
print("The reason why I love computing")
for step in range(21):
    # print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    print(step, sess.run([gradient, gvs], feed_dict={X: x_data, Y: y_data}), sess.run(W))
    sess.run(apply_gradients, feed_dict={X: x_data, Y: y_data})
    # sess.run(train, feed_dict={X: x_data, Y: y_data})
# W_val = []
# cost_val = []
# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict={W: [feed_W], X: x_data, Y: y_data})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)

# plt.plot(W_val, cost_val)
# plt.show()