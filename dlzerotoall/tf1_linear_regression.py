import tensorflow as tf

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# y = Wx + b
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost(loss) function cost(W, b) = 1/m sum(from i=1 to m) {H(x)_i-y(x)_i}^2
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#tf.reduce_mean 텐서의 평균 제공
"""
foobar [1., 2., 3., 4.]
tf.reduce_mean(foobar)
-> 2.5
"""

# cost minimization
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#session

sess = tf.Session()
#initializes global variables in the graph
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # sess.run(train)
    # train을 세션에서 실행할 때 리턴되는 value는 _에 저장
    # a, b, c, d = [1, 2, 3, 4] -> a: 1 , b:2 and so on...
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5], Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)
        # print(step, sess.run(cost), sess.run(W), sess.run(b))

print(sess.run(hypothesis, feed_dict={X: [5]}))
print(sess.run(hypothesis, feed_dict={X: [2.5]}))
print(sess.run(hypothesis, feed_dict={X: [1.5, 3.5]}))