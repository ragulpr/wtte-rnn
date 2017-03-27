import numpy as np
import tensorflow as tf
from keras import backend as K
# TODO works only with tf backend.
class output_activation(object):

    def __init__(self, init_alpha=1.0, max_beta_value=5.0):
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value

    def activation(self, ab):
        """Elementwise computation of alpha and regularized beta.

            Alpha: 
            (activation) 
            Exponential units seems to give faster training than 
            the original papers softplus units. Makes sense due to logarithmic
            effect of change in alpha. 
            (initialization) 
            To get faster training and fewer exploding gradients,
            initialize alpha to be around its scale when beta is around 1.0,
            approx the expected value/mean of training tte. 
            Because we're lazy we want the correct scale of output built
            into the model so initialize implicitly; 
            multiply assumed exp(0)=1 by scale factor `init_alpha`.

            Beta: 
            (activation) 
            We want slow changes when beta-> 0 so Softplus made sense in the original 
            paper but we get similar effect with sigmoid. It also has nice features.
            (regularization) Use max_beta_value to implicitly regularize the model
            (initialization) Fixed to begin moving slowly around 1.0

            Assumes tensorflow backend.

            Args:
                ab: tensor with last dimension having length 2
                    with ab[-1][0] = alpha, ab[-1][1] = beta

            Returns:
                A positive `Tensor` of same shape as input
        """

        a, b = tf.unstack(ab, num=2, axis=-1)

        # Implicitly initialize alpha:
        a = self.init_alpha * K.exp(a)

        m = self.max_beta_value
        if m > 1.337:  # something >1.0
            # shift to start around 1.0
            # assuming input is around 0.0
            _shift = np.log(m - 1.0)

            # change scale to be approx unchanged
            _div = 1.0 / m
            b = K.sigmoid((b - _shift) * _div)
        else:
            b = K.sigmoid(b)

        # Clipped sigmoid : has zero gradient at 0,1
        # Reduces the small tendency of instability after long training
        # by zeroing gradient. 
        b = m * tf.clip_by_value(b, 1e-6, 0.999999)

        ab = tf.stack([a, b], axis=-1)

        return ab


class loss(object):

    def __init__(self,
                 kind,
                 use_censoring=True,
                 use_weights=False,
                 regularize=False,
                 location=10.0,
                 growth=20.0):

        self.kind = kind
        self.use_censoring = use_censoring
        self.use_weights = use_weights

        self.regularize = regularize
        if regularize:
            self.location = location
            self.growth = growth

        if 1 + use_weights + use_censoring != 2:
            print('target/feature shape mismatch not yet supported, ' +
                  'keras will raise error. See :' +
                  ' _check_loss_and_target_compatibility ' +
                  'https://github.com/fchollet/keras/blob/master/keras/engine/training.py#L237'
                  )

    def loss_function(self, y_true, y_pred):
        def keras_split(y_true, y_pred, use_censoring, use_weights):
            """
                Everything is a hack around the y_true-y_pred paradigm.
            """

            a, b = tf.unstack(y_pred, num=2, axis=-1)
            y_true = tf.unstack(
                y_true, num=1 + use_censoring + use_weights, axis=-1)

            y = y_true[0]

            if use_censoring:
                u = y_true[1]
            else:
                u = 1.0

            if use_weights:
                weights = y_true[-1]
            else:
                weights = None

            return y, u, a, b, weights

        def loglik_discrete(y, u, a, b, epsilon=1e-35):
            hazard0 = K.pow((y + epsilon) / a, b)
            hazard1 = K.pow((y + 1.0) / a, b)

            loglikelihoods = u * \
                K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1
            return loglikelihoods

        def loglik_continuous(y, u, a, b, epsilon=1e-35):
            ya = (y + epsilon) / a
            logLikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
            return logLikelihoods

        def penalty_term(b, location, growth):
            scale = growth / location
            penalty = K.exp(scale * (b - location))
            return penalty

        def accumulate_loss(loglikelihoods, weights=None):
            # TODO scale?
            # problem if weights sum to 1 per seq ex.
            # then steplength is effected.
            if weights is None:
                loss = -1.0 * K.mean(loglikelihoods, axis=-1)
            else:
                # TODO: this is dot product
                #                loss = -1.0*K.sum(loglikelihoods*weights,axis=-1)/K.sum(weights,axis=-1)
                loss = -1.0 * K.sum(loglikelihoods * weights) / K.sum(weights)
            return loss

        y, u, a, b, weights = keras_split(
            y_true, y_pred, self.use_censoring, self.use_weights)

        if self.kind == 'discrete':
            loglikelihoods = loglik_discrete(y, u, a, b)
        else:
            loglikelihoods = loglik_continuous(y, u, a, b)

        if self.regularize:
            loglikelihoods = loglikelihoods + \
                penalty_term(b, self.location, self.growth)

        loss = accumulate_loss(loglikelihoods, weights)

        return loss
