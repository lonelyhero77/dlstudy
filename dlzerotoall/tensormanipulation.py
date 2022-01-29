import numpy as np
import tensorflow as tf
import pprint

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

t = np.array(
    [0., 1., 2., 3., 4., 5., 6.]
)

pp.pprint(t)

print(t.ndim) #rank, dimension
print(t.shape) #shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

f = np.array(
    [
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
    ]
)
pp.pprint(f)
print(f.ndim)
print(f.shape)

k = tf.constant(
    [1, 2, 3, 4]
)
print(tf.shape(k).eval())

l = tf.constant(
    [
        [1, 2],
        [3, 4]
    ]
)
print(tf.shape(l).eval())

p = tf.constant(
    [
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]
            ],
            [
                [13, 14, 15, 16],
                [17, 18, 19, 20],
                [21, 22, 23, 24]
            ]
        ]
    ]
)
print(tf.shape(p).eval())

mat1 = tf.constant(
    [
        [1., 2.],
        [3., 4.]
    ]
)
mat2 = tf.constant(
    [
        [1.],
        [2.]
    ]
)

print(tf.matmul(mat1, mat2).eval()) # matrix multiplication
print((mat1 * mat2).eval()) # broadcasting

mat3 = tf.constant(
    [
        [3., 3.]
    ]
)
mat4 = tf.constant(
    [
        [2., 2.]
    ]
)

print((mat3 + mat4).eval())

mat5 = tf.constant(
    [
        [1., 2.]
    ]
)
mat6 = tf.constant(3.)
print((mat5 + mat6).eval())

mat7 = tf.constant([[1., 2.]])
mat8 = tf.constant([3., 4.])
print((mat7 + mat8).eval())

mat9 = tf.constant([[1., 2.]])
mat10 = tf.constant([[3.], [4.]])
print((mat9 + mat10).eval())
