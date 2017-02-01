def weibull_logLik_continuous(a, b, y_, u_, name=None):
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
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of log-likelihoods of same shape as a, b, y_, u_
    """

    ya = tf.div(y_+1e-35, a)  # Small optimization y/a
    return(
        tf.mul(u_,
               tf.log(b)+tf.mul(b, tf.log(ya))
        )-tf.pow(ya, b)
    )

def weibull_logLik_discrete(a, b, y_, u_, name=None):
    """Returns element-wise Weibull censored discrete log-likelihood.
    
    Unit-discretized weibull log-likelihood. loss=-loglikelihood.
    All input values must be of same type and shape.

    Args:
        a:  alpha. Positive nonzero `Tensor` of type `float32`, `float64`.
        b:  beta.  Positive nonzero `Tensor` of type `float32`, `float64`.
        y_: time to event. Positive `Tensor` of type `float32`, `float64`.
        u_: indicator 0.0 if right censored, 1.0 if uncensored
            `Tensor` of type `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of log-likelihoods of same shape as a, b, y_, u_
    """

    with tf.name_scope(name):
        hazard0 = tf.pow(tf.div(y_+1e-35, a), b)  # 1e-9 safe, really
        hazard1 = tf.pow(tf.div(y_+1.0, a), b)
    return(tf.mul(u_, tf.log(tf.exp(hazard1-hazard0)-1.0))-hazard1)

def weibull_betapenalty(b, location = 10.0, growth=20.0, name=None):
    """Returns a positive penalty term exploding when beta approaches location.

    Adding this term to the loss may prevent overfitting and numerical instability
    of large values of beta (overconfidence). Remember that loss = -loglik+penalty

    Args:
        b:  beta.  Positive nonzero `Tensor` of type `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A positive `Tensor` of same shape as `b` being a penalty term
    """
    with tf.name_scope(name):
        scale = growth/location
        penalty_ = tf.exp(scale*(b-location))
    return(penalty_)