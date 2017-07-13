from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#import pytest

import numpy as np
import pandas as pd
from six.moves import xrange

from wtte.transforms import padded_to_df
from wtte.pipelines import data_pipeline

from wtte.data_generators import generate_random_df


def run_test(
        # '1' on the end dirty way to not pollute testing-namespace
        id_col1='id',
        abs_time_col1='time_int',
        column_names1=['event', 'int_column', 'double_column'],
        discrete_time1=True,
        pad_between_steps1=False,
        infer_seq_endtime1=False,
        time_sec_interval1=1,
        timestep_aggregation_dict1=None):
    np.random.seed(1)

    # Should fail randomly if unique_times = False since it reduces those
    # times.
    df = generate_random_df(n_seqs=2, max_seq_length=5, unique_times=True)

    # rename the abs_time_col to something new to spot assumptions.
    df.rename(columns={"dt": abs_time_col1,
                       'id': id_col1}, inplace=True)

    padded, padded_t, seq_ids, df_collapsed = \
        data_pipeline(df,
                      id_col=id_col1,
                      abs_time_col=abs_time_col1,
                      column_names=column_names1,
                      discrete_time=discrete_time1,
                      pad_between_steps=pad_between_steps1,
                      infer_seq_endtime=infer_seq_endtime1,
                      time_sec_interval=time_sec_interval1,
                      timestep_aggregation_dict=timestep_aggregation_dict1,
                      drop_last_timestep=False
                      )

    if pad_between_steps1:
        df_new = padded_to_df(padded, column_names1, [
            int, int, float], ids=seq_ids, id_col=id_col1, t_col='t_elapsed')
        df = df[[id_col1, 't_elapsed']+column_names1].reset_index(drop=True)
        pd.util.testing.assert_frame_equal(df, df_new)
    else:
        df_new = padded_to_df(padded, column_names1, [
            int, int, float], ids=seq_ids, id_col=id_col1, t_col='t_ix')
        df = df[[id_col1, 't_ix']+column_names1].reset_index(drop=True)

        pd.util.testing.assert_frame_equal(df, df_new)


class TestPipeline():

    def test_discrete_padded_pipeline(self):
        run_test(
            # '1' on the end dirty way to not pollute testing-namespace
            id_col1='idnewname',
            abs_time_col1='time_int',
            column_names1=['event', 'int_column', 'double_column'],
            discrete_time1=True,
            pad_between_steps1=True,
            infer_seq_endtime1=False,
            time_sec_interval1=1,
            timestep_aggregation_dict1=None)

    def test_discrete_unpadded_pipeline(self):
        run_test(
            # '1' on the end dirty way to not pollute testing-namespace
            id_col1='idnewname',
            abs_time_col1='time_int',
            column_names1=['event', 'int_column', 'double_column'],
            discrete_time1=True,
            pad_between_steps1=False,
            infer_seq_endtime1=False,
            time_sec_interval1=1,
            timestep_aggregation_dict1=None)

    def test_continuous_pipeline(self):
        run_test(
            # '1' on the end dirty way to not pollute testing-namespace
            id_col1='idnewname',
            abs_time_col1='time_int',
            column_names1=['event', 'int_column', 'double_column'],
            discrete_time1=True,
            pad_between_steps1=False,
            infer_seq_endtime1=False,
            time_sec_interval1=1,
            timestep_aggregation_dict1=None)

    # def test_discrete_time_continous():  # TODO
    # def test_continuous_time_discrete(): # TODO
    # TODO test with flag infer enddtime
    # TODO test with tte and censoring etc.

