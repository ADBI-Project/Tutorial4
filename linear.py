from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np

boston = load_boston()

d = len(boston['feature_names'])
N = len(boston['data'])
x_data = boston['data'].astype(np.float32)
y_data = boston['target'].reshape((-1, 1))

X = tf.placeholder(tf.float32, shape=[None, d], name='X')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y')

W = tf.Variable(tf.random_uniform([d, 1], -1.0, 1.0))
y = tf.matmul(X, W, name='y_pred')

loss = tf.reduce_mean(tf.square(y_ - y))
optimizer = tf.train.GradientDescentOptimizer(10e-7)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_batch = N
for step in range(1000):
    i_batch = (step % n_batch) 
    batch = x_data[i_batch:i_batch+1], y_data[i_batch:i_batch+1]
    sess.run(train, feed_dict={X: batch[0], y_: batch[1]})
    # if step % 20 == 0:
		    	
print sess.run(W)