from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def loglik_continuous(a, b, y_, u_, output_collection=(), name=None):
    """Returns element-wise Weibull censored log-likelihood.

    Continuous weibull log-likelihood. loss=-loglikelihood.
    All input values must be of same type and shape.

    Args:
        a:  alpha. Positive nonzero `Tensor` of type `float32`, `float64`.
        b:  beta.  Positive nonzero `Tensor` of type `float32`, `float64`.
        y_: time to event. Positive  nonzero `Tensor` of type `float32`, 
            `float64`.
        u_: indicator 0.0 if right censored, 1.0 if uncensored
            `Tensor` of type `float32`, `float64`.
        output_collection: `tuple` of `string`s, name of the collection to
                          collect result of this op.
        name: `string`, name of the operation.

    Returns:
        A `Tensor` of log-likelihoods of same shape as a, b, y_, u_
    """

    with tf.name_scope(name, "weibull_loglik_continuous", [a, b, y_, u_]):

        ya = tf.div(y_ + 1e-35, a)  # Small optimization y/a

        loglik = tf.multiply(u_, tf.log(
            b) + tf.multiply(b, tf.log(ya))) - tf.pow(ya, b)
        tf.add_to_collection(output_collection, loglik)

    return(loglik)


def loglik_discrete(a, b, y_, u_, output_collection=(), name=None):
    """Returns element-wise Weibull censored discrete log-likelihood.

    Unit-discretized weibull log-likelihood. loss=-loglikelihood.
    All input values must be of same type and shape.

    Args:
        a:  alpha. Positive nonzero `Tensor` of type `float32`, `float64`.
        b:  beta.  Positive nonzero `Tensor` of type `float32`, `float64`.
        y_: time to event. Positive `Tensor` of type `float32`, `float64`.
        u_: indicator 0.0 if right censored, 1.0 if uncensored
            `Tensor` of type `float32`, `float64`.
        output_collection: `tuple` of `string`s, name of the collection to
                          collect result of this op.
        name: `string`, name of the operation.
    Returns:
        A `Tensor` of log-likelihoods of same shape as a, b, y_, u_
    """

    with tf.name_scope(name, "weibull_loglik_discrete", [a, b, y_, u_]):
        hazard0 = tf.pow(tf.div(y_ + 1e-35, a), b)  # 1e-9 safe, really
        hazard1 = tf.pow(tf.div(y_ + 1.0, a), b)
        loglik = tf.multiply(u_, tf.log(
            tf.exp(hazard1 - hazard0) - 1.0)) - hazard1

        tf.add_to_collection(output_collection, loglik)
    return(loglik)


def betapenalty(b, location=10.0, growth=20.0, output_collection=(), name=None):
    """Returns a positive penalty term exploding when beta approaches location.

    Adding this term to the loss may prevent overfitting and numerical instability
    of large values of beta (overconfidence). Remember that loss = -loglik+penalty

    Args:
        b:  beta.  Positive nonzero `Tensor` of type `float32`, `float64`.
        output_collection: `tuple` of `string`s, name of the collection to
                          collect result of this op.
        name: `string`, name of the operation.

    Returns:
        A positive `Tensor` of same shape as `b` being a penalty term
    """
    with tf.name_scope(name, "weibull_betapenalty", [a, b, y_, u_]):
        scale = growth / location
        penalty = tf.exp(scale * (b - location))
        tf.add_to_collection(output_collection, penalty)

    return(penalty)
