from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import xrange

from .transforms import df_join_in_endtime
from .transforms import df_to_padded


def data_pipeline(
        df,
        id_col='id',
        abs_time_col='time_int',
        column_names=["event"],
        constant_cols=[],
        discrete_time=True,
        pad_between_steps=True,
        infer_seq_endtime=True,
        time_sec_interval=60 * 60 * 24,
        timestep_aggregation_dict=None,
        drop_last_timestep=True):
    """Preprocess dataframe and return it in padded tensor format.

        This function is due to change alot.

        1. Lowers the resolution of the (int) `abs_time_col` ex from epoch sec to epoch day by aggregating\
          each column using `timestep_aggregation_dict`.

        2. Padds out with zeros between timesteps and fills with value of `constant_cols`.

        3. Infers or adds/fills an endtime.

        This outputs tensor as is and leave it to downstream to define events, disalign targets
        and features (see `shift_discrete_padded_features`) and from that
        censoring-indicator and tte.
    """
    if timestep_aggregation_dict is None:
        timestep_aggregation_dict = dict.fromkeys(column_names, "sum")

    if discrete_time:
        # Lower resolution on unix-timestamp ex. to day
        df[abs_time_col] = time_sec_interval * \
            (df[abs_time_col] // (time_sec_interval))
        # Last timestep may be incomplete/not fully measured so drop it.
        if drop_last_timestep:
            df = df.loc[df[abs_time_col] < df[abs_time_col].max()]

    # COLLAPSE STRATEGY : SUM by default
    # Aggregate over the new datetime interval to get id,time_int = unique key
    # value pair

    df = df.groupby([id_col, abs_time_col], as_index=False).\
        agg(timestep_aggregation_dict)

    if infer_seq_endtime:
        # Assuming each sequence has its own start and is not terminated by last event:
        # Add last time that we knew the sequence was 'alive'.
        df = df_join_in_endtime(df,
                                constant_per_id_cols=[
                                    id_col] + constant_cols,
                                abs_time_col=abs_time_col)

        # this will cast every column with NaN to float
        df = df.fillna(0, inplace=False)

    df = df.sort_values([id_col, abs_time_col],
                        inplace=False).reset_index(drop=True)

    # Add "elapsed time" t_elapsed = 0,3,99,179,.. for each user.
    df['t_elapsed'] = df.groupby([id_col], group_keys=False).apply(
        lambda g: g[abs_time_col] - g[abs_time_col].min())

    if discrete_time:
        # Let each integer step stand for the time-resolution, eg. days:
        df['t_elapsed'] = df['t_elapsed'].astype(int) // time_sec_interval
    #########

    if pad_between_steps:
        # if we set 't_elapsed' as t_col we'll pad between observed steps in
        # df_to_padded
        t_col = 't_elapsed'
    else:
        # Add t_ix = 0,1,2,3,.. and set as primary user-time indicator.
        # if we set 't_ix' as t_col no padding between steps is done in
        # df_to_padded
        df['t_ix'] = df.groupby([id_col])['t_elapsed'].\
            rank(method='dense').astype(int) - 1
        t_col = 't_ix'

    # Here go over to to-tensor operations. Could be split into two funs

    # Everything to tensor.
    if pad_between_steps:
        t_col = 't_elapsed'
        # By default expands to 0,1,2,... which is fine since this is true for
        # padded values.
        padded_t = None
        if not discrete_time:
            raise ValueError(
                "pad_between_steps = True, discrete_time = False \
                \n Can't padd between on a continuous scale")
    else:
        # Add t_ix = 0,1,2,3,.. and set as primary user-time indicator.
        # if we set 't_ix' as t_col no padding between steps is done in
        # df_to_padded
        t_col = 't_ix'
        padded_t = df_to_padded(df, id_col=id_col, column_names=[
            't_elapsed'], t_col=t_col).squeeze()
        if discrete_time:
            padded_t = padded_t.astype(int)

    padded = df_to_padded(
        df, id_col=id_col, column_names=column_names, t_col=t_col)

    # If some columns where specified as constant
    if constant_cols is not None:
        if len(constant_cols) > 0:
            constant_cols_ix = []
            for i in xrange(len(column_names)):
                if column_names[i] in constant_cols:
                    constant_cols_ix.append(i)

            # Store away nan-mask to reapply it later.
            nan_mask = np.expand_dims(padded[:, :, 0], -1) * 0

            # Map data for first eventstep to every step
            padded[:, :, constant_cols_ix] = np.expand_dims(
                padded[:, 0, constant_cols_ix], 1)
            # Reapply nan-mask
            padded[:, :, constant_cols_ix] = padded[
                :, :, constant_cols_ix] + nan_mask

    seq_ids = df[id_col].unique()
    return padded, padded_t, seq_ids, df
