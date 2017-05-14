from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import tensorflow as tf
import numpy as np

from wtte.objectives.tensorflow import loglik_continuous, loglik_discrete

# SANITY CHECK: Use pure Weibull data censored at C(ensoring point).
# Should converge to the generating A(alpha) and B(eta) for each timestep


def generate_data(A, B, C, shape, discrete_time):
    # Generate Weibull random variables
    W = np.sort(A * np.power(-np.log(np.random.uniform(0, 1, shape)), 1 / B))

    if discrete_time:
        C = np.floor(C)
        W = np.floor(W)

    U = np.less_equal(W, C) * 1.
    Y = np.minimum(W, C)
    return W, Y, U

n_samples = 1000
n_features = 1
real_a = 3.
real_b = 2.
censoring_point = real_a * 2


def tf_loglik_runner(loglik_fun, discrete_time):
    sess = tf.Session()
    np.random.seed(1)
    tf.set_random_seed(1)

    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    u_ = tf.placeholder(tf.float32, shape=(None, 1))

    a = tf.exp(tf.Variable(tf.random_normal([1]), name='a_weight'))
    b = tf.exp(tf.Variable(tf.random_normal([1]), name='b_weight'))

    # testing part:
    loglik = loglik_fun(a, b, y_, u_)

    loss = -tf.reduce_mean(loglik)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    train_step = optimizer.minimize(loss)

    # Launch the graph in a session.
    np.random.seed(1)

    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    tte_actual, tte_censored, u_train = generate_data(
        A=real_a,
        B=real_b,
        C=censoring_point,  # <np.inf -> impose censoring
        shape=[n_samples, n_features],
        discrete_time=discrete_time)

    # Fit
    for step in range(1000):
        loss_val, _, a_val, b_val = sess.run([loss, train_step, a, b], feed_dict={
                                             y_: tte_censored, u_: u_train})

#        print('iteration:',step,'alpha :',a_val,'beta :',b_val,'discrete_time: ',discrete_time)
    print((real_a - a_val)**2, (real_b - b_val)**2)
    assert (real_a - a_val)**2 < 0.01
    assert (real_b - b_val)**2 < 0.01
    sess.close()
    tf.reset_default_graph()


def test_loglik_continuous():
    tf_loglik_runner(loglik_continuous, discrete_time=False)


def test_loglik_discrete():
    tf_loglik_runner(loglik_discrete, discrete_time=True)
