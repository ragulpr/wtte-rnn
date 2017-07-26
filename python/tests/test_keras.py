from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Lambda, Masking
from keras.layers.wrappers import TimeDistributed

from keras.optimizers import RMSprop

from wtte import wtte as wtte
from wtte.data_generators import generate_weibull


def test_keras_unstack_hack():
    y_true_np = np.random.random([1, 3, 2])
    y_true_np[:, :, 0] = 0
    y_true_np[:, :, 1] = 1

    y_true_keras = K.variable(y_true_np)

    y, u = wtte._keras_unstack_hack(y_true_keras)
    y_true_keras_new = K.stack([y, u], axis=-1)

    np.testing.assert_array_equal(K.eval(y_true_keras_new), y_true_np)

# SANITY CHECK: Use pure Weibull data censored at C(ensoring point).
# Should converge to the generating A(alpha) and B(eta) for each timestep


def get_data(discrete_time):
    y_test, y_train, u_train = generate_weibull(A=real_a,
                                                B=real_b,
                                                # <np.inf -> impose censoring
                                                C=censoring_point,
                                                shape=[n_sequences,
                                                       n_timesteps, 1],
                                                discrete_time=discrete_time)
    # With random input it _should_ learn weight 0
    x_train = x_test = np.random.uniform(
        low=-1, high=1, size=[n_sequences, n_timesteps, n_features])

    # y_test is uncencored data
    y_test = np.append(y_test, np.ones_like(y_test), axis=-1)
    y_train = np.append(y_train, u_train, axis=-1)
    return y_train, x_train, y_test, x_test


n_sequences = 10000
n_timesteps = 2
n_features = 1

real_a = 3.
real_b = 2.
censoring_point = real_a * 2

mask_value = -10.

lr = 0.02


def model_no_masking(discrete_time, init_alpha, max_beta):
    model = Sequential()
    model.add(TimeDistributed(Dense(2), input_shape=(n_timesteps, n_features)))

    model.add(Lambda(wtte.output_lambda, arguments={"init_alpha": init_alpha,
                                                    "max_beta_value": max_beta}))

    if discrete_time:
        loss = wtte.loss(kind='discrete').loss_function
    else:
        loss = wtte.loss(kind='continuous').loss_function

    model.compile(loss=loss, optimizer=RMSprop(lr=lr))

    return model


def model_masking(discrete_time, init_alpha, max_beta):
    model = Sequential()

    model.add(Masking(mask_value=mask_value,
                      input_shape=(n_timesteps, n_features)))
    model.add(TimeDistributed(Dense(2)))
    model.add(Lambda(wtte.output_lambda, arguments={"init_alpha": init_alpha,
                                                    "max_beta_value": max_beta}))

    if discrete_time:
        loss = wtte.loss(kind='discrete', reduce_loss=False).loss_function
    else:
        loss = wtte.loss(kind='continuous', reduce_loss=False).loss_function

    model.compile(loss=loss, optimizer=RMSprop(
        lr=lr), sample_weight_mode='temporal')
    return model


def keras_loglik_runner(discrete_time, add_masking):
    np.random.seed(1)
#    tf.set_random_seed(1)

    y_train, x_train, y_test, x_test = get_data(discrete_time=discrete_time)

    if add_masking:
        # If masking doesn't work, it'll learn nonzero weights (strong signal):
        x_train[:int(n_sequences / 2), int(n_timesteps / 2):, :] = mask_value
        y_train[:int(n_sequences / 2), int(n_timesteps / 2):, 0] = real_a * 300.

        weights = np.ones([n_sequences, n_timesteps])
        weights[:int(n_sequences / 2), int(n_timesteps / 2):] = 0.

        model = model_masking(
            discrete_time, init_alpha=real_a, max_beta=real_b * 3)
        model.fit(x_train, y_train,
                  epochs=5,
                  batch_size=100,
                  verbose=0,
                  sample_weight=weights,
                  )
    else:
        model = model_no_masking(
            discrete_time, init_alpha=real_a, max_beta=real_b * 3)

        model.fit(x_train, y_train,
                  epochs=5,
                  batch_size=100,
                  verbose=0,
                  )

    predicted = model.predict(x_test[:1, :, :])
    a_val = predicted[:, :, 0].mean()
    b_val = predicted[:, :, 1].mean()

    print(np.abs(real_a - a_val), np.abs(real_b - b_val))
    assert np.abs(real_a - a_val) < 0.1, 'alpha not converged'
    assert np.abs(real_b - b_val) < 0.1, 'beta not converged'


def test_loglik_continuous():
    keras_loglik_runner(discrete_time=False, add_masking=False)


def test_loglik_discrete():
    keras_loglik_runner(discrete_time=True, add_masking=False)


def test_loglik_continuous_masking():
    keras_loglik_runner(discrete_time=False, add_masking=True)


def test_output_lambda_initialization():
    # Initializing beta =1 gives us a simple initialization rule for alpha.
    # it also makes sense considering it initializes the hazard to flat
    # and in general as a regular exponential regression model.
    init_alpha = 5
    n_features = 1
    n_timesteps = 10
    n_sequences = 5
    np.random.seed(1)

    model = Sequential()

    # Identity layer
    model.add(Lambda(lambda x: x, input_shape=(n_timesteps, n_features)))
    model.add(Dense(2))
    model.add(Lambda(wtte.output_lambda,
                     arguments={"init_alpha": init_alpha,
                                "max_beta_value": 4.,
                                "alpha_kernel_scalefactor": 1.
                                }))

    # Test
    x = np.random.normal(0, 0.01, size=[n_sequences, n_timesteps, n_features])
    predicted = model.predict(x)

    # Check that it initializes +- 0.1 from init_alpha,beta=1
    abs_error = np.abs(predicted[:, :, 0].flatten() - init_alpha).mean()
    assert abs_error < 0.1, 'alpha initialization error' + str(abs_error)
    abs_error = np.abs(predicted[:, :, 1].flatten() - 1.).mean()
    assert abs_error < 0.1, 'beta initialization error' + str(abs_error)
