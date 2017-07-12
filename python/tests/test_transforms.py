from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import numpy as np
import pandas as pd
from six.moves import xrange

from wtte.transforms import df_to_padded
from wtte.transforms import padded_to_df
from wtte.transforms import shift_discrete_padded_features
from wtte.transforms import left_pad_to_right_pad
from wtte.transforms import right_pad_to_left_pad


def generate_random_df(n_seqs,
                       max_seq_length,
                       unique_times=True,
                       starttimes_min=0,
                       starttimes_max=0):
    """ generates random dataframe for testing.
    :param df: pandas dataframe with columns
      * `id`: integer
      * `t`: integer
      * `dt`: integer mimmicking a global event time
      * `t_ix`: integer contiguous user time count per id 0,1,2,..
      * `t_elapsed`: integer the time from starttime per id ex 0,1,10,..
      * `event`: 0 or 1
      * `int_column`: random data
      * `double_column`: dandom data
    :param unique_times: whether there id,elapsed_time has only one obs. Default true
    :param starttimes_min: integer to generate `dt` the absolute time
    :param starttimes_max: integer to generate `dt` the absolute time
    """

    seq_lengths = np.random.randint(max_seq_length, size=n_seqs) + 1
    id_list = []
    t_list = []
    dt_list = []

    if starttimes_min < starttimes_max:
        starttimes = np.sort(np.random.randint(
            low=starttimes_min, high=starttimes_max, size=n_seqs))
    else:
        starttimes = np.zeros(n_seqs)

    for s in xrange(n_seqs):
        # Each sequence is consists of n_obs in the range 0-seq_lengths[s]
        n_obs = np.sort(np.random.choice(
            seq_lengths[s], 1, replace=False)) + 1

        # Each obs occurred at random times
        t_elapsed = np.sort(np.random.choice(
            seq_lengths[s], n_obs, replace=not unique_times))

        # there's always an obs at the assigned first and last timestep for
        # this seq
        if seq_lengths[s] - 1 not in t_elapsed:
            t_elapsed = np.append(t_elapsed, seq_lengths[s] - 1)
        if 0 not in t_elapsed:
            t_elapsed = np.append(t_elapsed, 0)

        t_elapsed = np.sort(t_elapsed)

        id_list.append(np.repeat(s, repeats=len(t_elapsed)))
        dt_list.append(starttimes[s] + t_elapsed)
        t_list.append(t_elapsed)

    # unlist to one array
    id_column = [item for sublist in id_list for item in sublist]
    dt_column = [item for sublist in dt_list for item in sublist]
    t_column = [item for sublist in t_list for item in sublist]
    del id_list, dt_list, t_list

    # do not assume row indicates event!
    event_column = np.random.randint(2, size=len(t_column))
    int_column = np.arange(len(event_column)).astype(int)
    double_column = np.random.uniform(high=1, low=0, size=len(t_column))

    df = pd.DataFrame({'id': id_column,
                       'dt': dt_column,
                       't_elapsed': t_column,
                       'event': event_column,
                       'int_column': int_column,
                       'double_column': double_column
                       })

    df['t_ix'] = df.groupby(['id'])['t_elapsed'].rank(
        method='dense').astype(int) - 1
    df = df[['id', 'dt', 't_ix', 't_elapsed',
             'event', 'int_column', 'double_column']]
    df = df.reset_index(drop=True)

    return df


class TestDfToPaddedPaddedToDf:
    """tests df_to_padded and padded_to_df
    generates a dataframe, transforms it to tensor format then back
    to the same df.
    """

    def record_based():
        """here only
        """
        n_seqs = 5
        max_seq_length = 10
        ids = xrange(n_seqs)
        cols_to_expand = ['event', 'int_column', 'double_column']
        np.random.seed(1)

        df = generate_random_df(n_seqs, max_seq_length)
        df['t_ix'] = df.groupby(['id'])['t_elapsed'].rank(
            method='dense').astype(int) - 1
        df = df.reset_index(drop=True)

        # Column names to transform to tensor
        dtypes = df[cols_to_expand].dtypes.values
        padded = df_to_padded(df, cols_to_expand, 'id', 't_ix')

        df_new = padded_to_df(padded, cols_to_expand,
                              dtypes, ids, 'id', 't_ix')
        # Pandas is awful. Index changes when slicing
        df = df[['id', 't_ix'] + cols_to_expand].reset_index(drop=True)
        pd.util.testing.assert_frame_equal(df, df_new)

    def padded_between():
        """Tests df_to_padded, padded_to_df
        """
        n_seqs = 5
        max_seq_length = 10
        ids = xrange(n_seqs)
        cols_to_expand = ['event', 'int_column', 'double_column']
        np.random.seed(1)

        df = generate_random_df(n_seqs, max_seq_length)
        df = df.reset_index(drop=True)

        # Column names to transform to tensor
        dtypes = df[cols_to_expand].dtypes.values
        padded = df_to_padded(df, cols_to_expand, 'id', 't_elapsed')

        df_new = padded_to_df(padded, cols_to_expand,
                              dtypes, ids, 'id', 't_elapsed')
        # Pandas is awful. Index changes when slicing
        df = df[['id', 't_elapsed'] + cols_to_expand].reset_index(drop=True)
        pd.util.testing.assert_frame_equal(df, df_new)


def test_shift_discrete_padded_features():
    """test for `discrete_padded_features`.
        TODO buggy unit. Due to change.
    """
    x = np.array([[[1], [2], [3]]])
    assert x.shape == (1, 3, 1)

    x_shifted = shift_discrete_padded_features(x, fill=0)

    np.testing.assert_array_equal(x_shifted, np.array([[[0], [1], [2]]]))


def test_align_padded():
    """test the function for switching between left-right padd.
    """
    # For simplicity, create a realistic padded tensor
    np.random.seed(1)
    n_seqs = 10
    max_seq_length = 10
    df = generate_random_df(n_seqs, max_seq_length)
    cols_to_expand = ['event', 'int_column', 'double_column']
    padded = df_to_padded(df, cols_to_expand, 'id', 't_elapsed')

    np.testing.assert_array_equal(
        padded, left_pad_to_right_pad(right_pad_to_left_pad(padded)))
    padded = np.copy(np.squeeze(padded))
    np.testing.assert_array_equal(
        padded, left_pad_to_right_pad(right_pad_to_left_pad(padded)))
