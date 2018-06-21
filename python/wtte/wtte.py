from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import log

import warnings
import numpy as np

from keras import backend as K
from keras.callbacks import Callback


def _keras_unstack_hack(ab):
    """Implements tf.unstack(y_true_keras, num=2, axis=-1).

       Keras-hack adopted to be compatible with Theano backend.

       :param ab: stacked variables
       :return a, b: unstacked variables
    """
    ndim = len(K.int_shape(ab))
    if ndim == 0:
        print('can not unstack with ndim=0')
    else:
        a = ab[..., 0]
        b = ab[..., 1]
    return a, b


def output_lambda(x, init_alpha=1.0, max_beta_value=5.0, scalefactor=None,
                  alpha_kernel_scalefactor=None):
    """Elementwise (Lambda) computation of alpha and regularized beta.

        - Alpha:

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

        - Beta:

            (activation)
            We want slow changes when beta-> 0 so Softplus made sense in the original
            paper but we get similar effect with sigmoid. It also has nice features.
            (regularization) Use max_beta_value to implicitly regularize the model
            (initialization) Fixed to begin moving slowly around 1.0

        - Usage
            .. code-block:: python

                model.add(TimeDistributed(Dense(2)))
                model.add(Lambda(wtte.output_lambda, arguments={"init_alpha":init_alpha, 
                                                        "max_beta_value":2.0
                                                       }))


        :param x: tensor with last dimension having length 2 with x[...,0] = alpha, x[...,1] = beta
        :param init_alpha: initial value of `alpha`. Default value is 1.0.
        :param max_beta_value: maximum beta value. Default value is 5.0.
        :param max_alpha_value: maxumum alpha value. Default is `None`.
        :type x: Array
        :type init_alpha: Float
        :type max_beta_value: Float
        :type max_alpha_value: Float
        :return x: A positive `Tensor` of same shape as input
        :rtype: Array

    """
    if max_beta_value is None or max_beta_value > 3:
        if K.epsilon() > 1e-07 and K.backend() == 'tensorflow':
            # TODO need to think this through lol
            message = "\
            Using tensorflow backend and allowing high `max_beta_value` may lead to\n\
            gradient NaN during training unless `K.epsilon()` is small.\n\
            Call `keras.backend.set_epsilon(1e-08)` to lower epsilon \
            "
            warnings.warn(message)
    if alpha_kernel_scalefactor is not None:
        message = "`alpha_kernel_scalefactor` deprecated in favor of `scalefactor` scaling both.\n Setting `scalefactor = alpha_kernel_scalefactor`"
        warnings.warn(message)
        scalefactor = alpha_kernel_scalefactor

    a, b = _keras_unstack_hack(x)

    if scalefactor is not None:
        # Done after due to theano bug.
        a, b = scalefactor * a, scalefactor * b

    # Implicitly initialize alpha:
    a = init_alpha * K.exp(a)

    if max_beta_value > 1.05:  # some value >>1.0
        # shift to start around 1.0
        # assuming input is around 0.0
        _shift = np.log(max_beta_value - 1.0)

        b = b - _shift

    b = max_beta_value * K.sigmoid(b)

    x = K.stack([a, b], axis=-1)

    return x


class OuputActivation(object):
    """ Elementwise computation of alpha and regularized beta.


        Wrapper to `output_lambda` using keras.layers.Activation. 
        See this for details.

        - Usage
            .. code-block:: python

               wtte_activation = wtte.OuputActivation(init_alpha=1.,
                                                 max_beta_value=4.0).activation

               model.add(Dense(2))
               model.add(Activation(wtte_activation))

    """

    def __init__(self, init_alpha=1.0, max_beta_value=5.0):
        self.init_alpha = init_alpha
        self.max_beta_value = max_beta_value

    def activation(self, ab):
        """ (Internal function) Activation wrapper

        :param ab: original tensor with alpha and beta.
        :return ab: return of `output_lambda` with `init_alpha` and `max_beta_value`.
        """
        ab = output_lambda(ab, init_alpha=self.init_alpha,
                           max_beta_value=self.max_beta_value)

        return ab

# Backwards Compatibility
output_activation = OuputActivation


def _keras_split(y_true, y_pred):
    """
        Everything is a hack around the y_true,y_pred paradigm.
    """
    y, u = _keras_unstack_hack(y_true)
    a, b = _keras_unstack_hack(y_pred)

    return y, u, a, b

keras_split = _keras_split


def loglik_discrete(y, u, a, b, epsilon=K.epsilon()):
    hazard0 = K.pow((y + epsilon) / a, b)
    hazard1 = K.pow((y + 1.0) / a, b)

    loglikelihoods = u * \
        K.log(K.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1
    return loglikelihoods


def loglik_continuous(y, u, a, b, epsilon=K.epsilon()):
    ya = (y + epsilon) / a
    loglikelihoods = u * (K.log(b) + b * K.log(ya)) - K.pow(ya, b)
    return loglikelihoods


def loglik_continuous_conditional_correction(y, u, a, b, epsilon=K.epsilon()):
    """Integrated conditional excess loss.
        Explanation TODO
    """
    ya = (y + epsilon) / a
    loglikelihoods = y * \
        (u * (K.log(b) + b * K.log(ya)) - (b / (b + 1.)) * K.pow(ya, b))
    return loglikelihoods


class Loss(object):
    """ Creates a keras WTTE-loss function.
        - Usage

            :Example:

            .. code-block:: python
               loss = wtte.Loss(kind='discrete').loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01))

               # And with masking:
               loss = wtte.Loss(kind='discrete',reduce_loss=False).loss_function
               model.compile(loss=loss, optimizer=RMSprop(lr=0.01),
                              sample_weight_mode='temporal')

        .. note::

            With masking keras needs to access each loss-contribution individually.
            Therefore we do not sum/reduce down to scalar (dim 1), instead return a 
            tensor (with reduce_loss=False).

        :param kind:  One of 'discrete' or 'continuous'
        :param reduce_loss: 
        :param clip_prob: Clip likelihood to [log(clip_prob),log(1-clip_prob)]
        :param regularize: Deprecated.
        :param location: Deprecated.
        :param growth: Deprecated.
        :type reduce_loss: Boolean
    """

    def __init__(self,
                 kind,
                 reduce_loss=True,
                 clip_prob=1e-6,
                 regularize=False,
                 location=None,
                 growth=None):

        self.kind = kind
        self.reduce_loss = reduce_loss
        self.clip_prob = clip_prob

        if regularize == True or location is not None or growth is not None:
            raise DeprecationWarning('Directly penalizing beta has been found \
                                      to be unneccessary when using bounded activation \
                                      and clipping of log-likelihood.\
                                      Use this method instead.')

    def loss_function(self, y_true, y_pred):

        y, u, a, b = _keras_split(y_true, y_pred)
        if self.kind == 'discrete':
            loglikelihoods = loglik_discrete(y, u, a, b)
        elif self.kind == 'continuous':
            loglikelihoods = loglik_continuous(y, u, a, b)

        if self.clip_prob is not None:
            loglikelihoods = K.clip(loglikelihoods, 
                log(self.clip_prob), log(1 - self.clip_prob))
        if self.reduce_loss:
            loss = -1.0 * K.mean(loglikelihoods, axis=-1)
        else:
            loss = -loglikelihoods

        return loss

# For backwards-compatibility
loss = Loss


class WeightWatcher(Callback):
    """Keras Callback to keep an eye on output layer weights.
        (under development)

        Usage:
            weightwatcher = WeightWatcher(per_batch=True,per_epoch=False)
            model.fit(...,callbacks=[weightwatcher])
            weightwatcher.plot()
    """

    def __init__(self,
                 per_batch=False,
                 per_epoch=True
                 ):
        self.per_batch = per_batch
        self.per_epoch = per_epoch

    def on_train_begin(self, logs={}):
        self.a_weights_mean = []
        self.b_weights_mean = []
        self.a_weights_min = []
        self.b_weights_min = []
        self.a_weights_max = []
        self.b_weights_max = []
        self.a_bias = []
        self.b_bias = []

    def append_metrics(self):
        # Last two weightlayers in model

        output_weights, output_biases = self.model.get_weights()[-2:]

        a_weights_mean, b_weights_mean = output_weights.mean(0)
        a_weights_min, b_weights_min = output_weights.min(0)
        a_weights_max, b_weights_max = output_weights.max(0)

        a_bias, b_bias = output_biases

        self.a_weights_mean.append(a_weights_mean)
        self.b_weights_mean.append(b_weights_mean)
        self.a_weights_min.append(a_weights_min)
        self.b_weights_min.append(b_weights_min)
        self.a_weights_max.append(a_weights_max)
        self.b_weights_max.append(b_weights_max)
        self.a_bias.append(a_bias)
        self.b_bias.append(b_bias)

    def on_train_end(self, logs={}):
        if self.per_epoch:
            self.append_metrics()
        return

    def on_epoch_begin(self, epoch, logs={}):
        if self.per_epoch:
            self.append_metrics()
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
        if self.per_batch:
            self.append_metrics()
        return

    def on_batch_end(self, batch, logs={}):
        if self.per_batch:
            self.append_metrics()
        return

    def plot(self):
        import matplotlib.pyplot as plt

        # Create axes
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.a_bias, color='b')
        ax1.set_xlabel('step')
        ax1.set_ylabel('alpha')

        ax2.plot(self.b_bias, color='r')
        ax2.set_ylabel('beta')

        # Change color of each axis
        def color_y_axis(ax, color):
            """Color your axes."""
            for t in ax.get_yticklabels():
                t.set_color(color)
            return None

        plt.title('biases')
        color_y_axis(ax1, 'b')
        color_y_axis(ax2, 'r')
        plt.show()

        ###############
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.plot(self.a_weights_min, color='blue',
                 linestyle='dotted', label='min', linewidth=2)
        ax1.plot(self.a_weights_mean, color='blue',
                 linestyle='solid', label='mean', linewidth=1)
        ax1.plot(self.a_weights_max, color='blue',
                 linestyle='dotted', label='max', linewidth=2)

        ax1.set_xlabel('step')
        ax1.set_ylabel('alpha')

        ax2.plot(self.b_weights_min, color='red',
                 linestyle='dotted', linewidth=2)
        ax2.plot(self.b_weights_mean, color='red',
                 linestyle='solid', linewidth=1)
        ax2.plot(self.b_weights_max, color='red',
                 linestyle='dotted', linewidth=2)
        ax2.set_ylabel('beta')

        # Change color of each axis
        def color_y_axis(ax, color):
            """Color your axes."""
            for t in ax.get_yticklabels():
                t.set_color(color)
            return None

        plt.title('weights (min,mean,max)')
        color_y_axis(ax1, 'b')
        color_y_axis(ax2, 'r')
        plt.show()
