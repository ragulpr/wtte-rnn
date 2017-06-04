from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import numpy as np
import pandas as pd

from wtte.transforms import *

try:
    xrange
except NameError:
    xrange = range


def generate_random_df(n_seqs, max_seq_length):
    """ generates random dataframe for testing.
    """

    seq_lengths = np.random.randint(max_seq_length, size=n_seqs) + 1
    t_list = []
    id_list = []
    dt_list = []

    for s in xrange(n_seqs):
        random_length = np.sort(np.random.choice(
            seq_lengths[s], 1, replace=False)) + 1
        t = np.sort(np.random.choice(
            seq_lengths[s], random_length, replace=False))

        if seq_lengths[s] - 1 not in t:
            t = np.insert(t, -1, seq_lengths[s] - 1)
        if 0 not in t:
            t = np.insert(t, 0, 0)

        t = np.sort(t)

        t_list.append(t)
#        dt_list.append(max_seq_length-seq_lengths[s]+ t)
        id_list.append(np.repeat(s, repeats=len(t)))

    id_column = [item for sublist in id_list for item in sublist]
    t_column = [item for sublist in t_list for item in sublist]
 #   dt_column      = [item for sublist in dt_list for item in sublist]

    # do not assume row indicates event!
    event_column = np.random.randint(2, size=len(t_column))
    int_column = np.arange(len(event_column)).astype(int)
    double_column = np.random.uniform(high=1, low=0, size=len(t_column))

    df = pd.DataFrame({'id': id_column,
                       't': t_column,
                       'event': event_column,
                       #                         'dt' : dt_column,
                       'int_column': int_column,
                       'double_column': double_column
                       })

#    df['dt']=df.groupby(['id'], group_keys=False).apply(lambda g: g.t.max())
    df = df.assign(dt=10 * df.id + df.t)
    df = df[['id', 't', 'dt', 'event', 'int_column', 'double_column']]
    return df


def test_df_to_padded_padded_to_df():
    """Tests df_to_padded, padded_to_df
    """

    # Call with names? Call with order?
    # Continuous tte?
    # Contiguous t?
    #
    np.random.seed(1)
    n_seqs = 100
    max_seq_length = 100
    ids = xrange(n_seqs)
    df = generate_random_df(n_seqs, max_seq_length)

    column_names = ['event', 'int_column', 'double_column']
    dtypes = ['double', 'int', 'float']

    padded = df_to_padded(df, column_names)

    df_new = padded_to_df(padded, column_names, dtypes, ids=ids)

    assert False not in (
        df[['id', 't', 'event', 'int_column', 'double_column']].values == df_new.values)[0],\
        'test_df_to_padded_padded_to_df failed'


def test_shift_discrete_padded_features():
    x = np.array([[[1], [2], [3]]])
    assert x.shape == (1, 3, 1)

    x_shifted = shift_discrete_padded_features(x, fill=0)

    np.testing.assert_array_equal(x_shifted, np.array([[[0], [1], [2]]]))


def test_align_padded():
    np.random.seed(1)
    n_seqs = 10
    max_seq_length = 10
    ids = xrange(n_seqs)
    df = generate_random_df(n_seqs, max_seq_length)

    column_names = ['event', 'int_column', 'double_column']
    dtypes = ['double', 'int', 'float']

    padded = df_to_padded(df, column_names)

    np.testing.assert_array_equal(
        padded, left_pad_to_right_pad(right_pad_to_left_pad(padded)))
    padded = np.copy(np.squeeze(padded))
    np.testing.assert_array_equal(
        padded, left_pad_to_right_pad(right_pad_to_left_pad(padded)))
