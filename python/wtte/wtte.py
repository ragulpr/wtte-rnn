from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K

try:
    xrange
except NameError:
    xrange = range


def _keras_unstack_hack(ab):
    """Implements tf.unstack(y_true_keras, num=2, axis=-1).
       Keras-hack adopted to be compatible with theano backend.
    """
    ndim = len(K.int_shape(ab))
    if ndim == 0:
        print('can not unstack with ndim=0')
    else:
        a = ab[..., 0]
        b = ab[..., 1]
    return a, b


def output_lambda(x, init_alpha=1.0, max_beta_value=5.0, max_alpha_value=None):
    """Elementwise (Lambda) computation of alpha and regularized beta.

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
            x: tensor with last dimension having length 2
                with x[...,0] = alpha, x[...,1] = beta

        Usage:
            model.add(Dense(2))
            model.add(Lambda(output_lambda, arguments={"init_alpha":100., "max_beta_value":2.0}))
        Returns:
            A positive `Tensor` of same shape as input
    """
    a, b = _keras_unstack_hack(x)

    # Implicitly initialize alpha:
    if max_alpha_value is None:
        a = init_alpha * K.exp(a)
    else:
        a = init_alpha * K.clip(x=a, min_value=K.epsilon(),
                                max_value=max_alpha_value)

    m = max_beta_value
    if m > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(m - 1.0)

        b = K.sigmoid(b - _shift)
    else:
        b = K.sigmoid(b)

    # Clipped sigmoid : has zero gradient at 0,1
    # Reduces the small tendency of instability after long training
    # by zeroing gradient.
    b = m * K.clip(x=b, min_value=K.epsilon(), max_value=1. - K.epsilon())

    x = K.stack([a, b], axis=-1)

    return x


class output_activation(object):
    """ Elementwise computation of alpha and regularized beta using keras.layers.Activation.
        Wrapper

        Usage:
            wtte_activation = wtte.output_activation(init_alpha=1.,
                                             max_beta_value=4.0).activation

            model.add(Dense(2))
            model.add(Activation(wtte_activation))

    """

    def __init__(self, init_alpha=1.0, max_beta_value=5.0):
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value

    def activation(self, ab):
        ab = output_lambda(ab, init_alpha=self.init_alpha,
                           max_beta_value=self.max_beta_value)

        return ab


class loss(object):
    """ Creates a keras WTTE-loss function.
        If regularize is called, a penalty is added creating 'wall' that beta do not
        want to pass over. This is not necessary with Sigmoid-beta activation.

        With masking keras needs to access each loss-contribution individually. Therefore
        we do not sum/reduce down to dim 1, instead a return tensor (with reduce_loss=False).

        Usage:
            loss = wtte.loss(kind='discrete').loss_function
            model.compile(loss=loss, optimizer=RMSprop(lr=0.01))
            And with masking:
            loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function
            model.compile(loss=loss, optimizer=RMSprop(lr=0.01),sample_weight_mode='temporal')

    """

    def __init__(self,
                 kind,
                 reduce_loss=True,
                 regularize=False,
                 location=10.0,
                 growth=20.0):

        self.kind = kind
        self.reduce_loss = reduce_loss

        self.regularize = regularize
        if regularize:
            self.location = location
            self.growth = growth

    def loss_function(self, y_true, y_pred):
        def keras_split(y_true, y_pred):
            """
                Everything is a hack around the y_true,y_pred paradigm.
            """
            y, u = _keras_unstack_hack(y_true)
            a, b = _keras_unstack_hack(y_pred)

            return y, u, a, b

        def loglik_discrete(y, u, a, b, epsilon=1e-35):
            hazard0 = K.pow((y + epsilon) / a, b)
            hazard1 = K.pow((y + 1.0) / a, b)

            loglikelihoods = u * \
                K.log(K.exp(hazard1 - hazard0) - 1.0) - hazard1
            return loglikelihoods

        def loglik_continuous(y, u, a, b, epsilon=1e-35):
            ya = (y + epsilon) / a
            loglikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
            return loglikelihoods

        def loglik_continuous_conditional_correction(y, u, a, b, epsilon=1e-35):
            """Integrated conditional excess loss.
                Explanation TODO
            """
            ya = (y + epsilon) / a
            loglikelihoods = y * \
                (u * (K.log(b) + b * K.log(ya)) - (b / (b + 1.)) * K.pow(ya, b))
            return loglikelihoods

        def penalty_term(b, location, growth):
            scale = growth / location
            penalty = K.exp(scale * (b - location))
            return penalty

        def accumulate_loss(loglikelihoods):
            loss = -1.0 * K.mean(loglikelihoods, axis=-1)
            return loss

        y, u, a, b = keras_split(y_true, y_pred)

        if self.kind == 'discrete':
            loglikelihoods = loglik_discrete(y, u, a, b)
        elif self.kind == 'continuous':
            loglikelihoods = loglik_continuous(y, u, a, b)

        if self.regularize:
            loglikelihoods = loglikelihoods + \
                penalty_term(b, self.location, self.growth)

        if self.reduce_loss:
            loss = accumulate_loss(loglikelihoods)
        else:
            loss = -loglikelihoods

        return loss
