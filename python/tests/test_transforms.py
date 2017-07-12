from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pytest

import numpy as np
import pandas as pd
from six.moves import xrange

from .util import generate_random_df

from wtte.transforms import df_to_padded
from wtte.transforms import padded_to_df
from wtte.transforms import shift_discrete_padded_features
from wtte.transforms import left_pad_to_right_pad
from wtte.transforms import right_pad_to_left_pad

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
