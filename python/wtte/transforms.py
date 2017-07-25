from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import xrange

from .tte_util import get_is_not_censored
from .tte_util import get_tte

def get_padded_seq_lengths(padded):
    """Returns the number of (seq_len) non-nan elements per sequence.

    :param padded: 2d or 3d tensor with dim 2 the time dimension
    """
    if len(padded.shape) == 2:
        # (n_seqs,n_timesteps)
        seq_lengths = np.count_nonzero(~np.isnan(padded), axis=1)
    elif len(padded.shape) == 3:
        # (n_seqs,n_timesteps,n_features,..)
        seq_lengths = np.count_nonzero(~np.isnan(padded[:, :, 0]), axis=1)
    else:
        print('not yet implemented')
        # TODO

    return seq_lengths

def df_to_array(df, column_names, nanpad_right=True, return_lists=False,
                id_col='id', t_col='t'):
    """Converts flat pandas df with cols `id,t,col1,col2,..` to array indexed `[id,t,col]`. 

    :param df: dataframe with columns:

      * `id`: Any type. A unique key for the sequence.

      * `t`: integer. If `t` is a non-contiguous int vec per id then steps in
        between t's are padded with zeros.

      * `columns` in `column_names` (String list)

    :type df: Pandas dataframe

    :param Boolean nanpad_right: If `True`, sequences are `np.nan`-padded to `max_seq_len`
    :param return_lists: Put every tensor in its own subarray
    :param id_col: string column name for `id`
    :param t_col: string column name for `t`
    :return padded: With seqlen the max value of `t` per id

      * if nanpad_right & !return_lists:
        a numpy float array of dimension `[n_seqs,max_seqlen,n_features]`

      * if nanpad_right & return_lists:
        n_seqs numpy float sub-arrays of dimension `[max_seqlen,n_features]`

      * if !nanpad_right & return_lists:
        n_seqs numpy float sub-arrays of dimension `[seqlen,n_features]`
    """

    # Do not sort. Create a view.
    grouped = df.groupby(id_col, sort=False)

    unique_ids = list(grouped.groups.keys())

    n_seqs = grouped.ngroups
    n_features = len(column_names)
    seq_lengths = df[[id_col, t_col]].groupby(
        id_col).aggregate('max')[t_col].values + 1

    # We can't assume to fit varying length seqs. in flat array without
    # padding.
    assert nanpad_right or len(
        set(seq_lengths)) == 1 or return_lists, 'Wont fit in flat array'

    max_seq_len = seq_lengths.max()

    # Initialize the array to be filled
    if return_lists:
        if nanpad_right:
            padded = np.split(np.zeros([n_seqs * max_seq_len, n_features]),
                              np.cumsum(np.repeat(max_seq_len, n_seqs)))
        else:
            padded = np.split(np.zeros([sum(seq_lengths), n_features]),
                              np.cumsum(seq_lengths))
    else:
        padded = np.zeros([n_seqs, max_seq_len, n_features])

    # Fill it
    for s in xrange(n_seqs):
        # df_user is a view
        df_group = grouped.get_group(unique_ids[s])

        padded[s][df_group[t_col].values, :] = df_group[column_names].values
        if nanpad_right and seq_lengths[s] < max_seq_len:
            padded[s][seq_lengths[s]:, :].fill(np.nan)

    return padded


def df_to_padded(df, column_names, id_col='id', t_col='t'):
    """Pads pandas df to a numpy array of shape `[n_seqs,max_seqlen,n_features]`.
        see `df_to_array` for details
    """
    return df_to_array(df, column_names, nanpad_right=True,
                       return_lists=False, id_col=id_col, t_col=t_col)


def df_to_subarrays(df, column_names, id_col='id', t_col='t'):
    """Pads pandas df to subarrays of shape `[n_seqs][seqlen[s],n_features]`.
        see `df_to_array` for details
    """
    return df_to_array(df, column_names, nanpad_right=False,
                       return_lists=True, id_col=id_col, t_col=t_col)


def padded_to_df(padded, column_names, dtypes, ids=None, id_col='id', t_col='t'):
    """Takes padded numpy array and converts nonzero entries to pandas dataframe row.

    Inverse to df_to_padded.

    :param Array padded: a numpy float array of dimension `[n_seqs,max_seqlen,n_features]`.
    :param list column_names: other columns to expand from df
    :param list dtypes:  the type to cast the float-entries to.
    :type dtypes: String list
    :param ids: (optional) the ids to attach to each sequence
    :param id_col: Column where `id` is located. Default value is `id`.
    :param t_col: Column where `t` is located. Default value is `t`.

    :return df: Dataframe with Columns

      *  `id` (Integer) or the value of `ids`

      *  `t` (Integer).

      A row in df is the t'th event for a `id` and has columns from `column_names`
    """

    def get_is_nonempty_mask(padded):
        """ (internal function) Non-empty masks

        :return is_nonempty: True if `[i,j,:]` has non-zero non-nan - entries or
            j is the start or endpoint of a sequence i
        :type is_nonempty: Boolean Array
        """

        # If any nonzero element then nonempty:
        is_nonempty = (padded != 0).sum(2) != 0

        # first and last step in each seq is not empty
        # (has info about from and to)
        is_nonempty[:, 0] = True

        seq_lengths = get_padded_seq_lengths(padded)

        is_nonempty[xrange(n_seqs), seq_lengths - 1] = True

        # nan-mask is always empty:
        is_nonempty[np.isnan(padded.sum(2))] = False

        return is_nonempty

    def get_basic_df(padded, is_nonempty, ids):
        """ (internal function) Get basic dataframe

        :return df: Dataframe with columns
          * `id`: a column of id
          * `t`: a column of (user/sequence) timestep
        """
        n_nonempty_steps = is_nonempty.sum(1)
        df = pd.DataFrame(index=xrange(sum(n_nonempty_steps)))

        id_vec = []
        for s in xrange(n_seqs):
            for reps in xrange(n_nonempty_steps[s]):
                id_vec.append(ids[s])

        df[id_col] = id_vec
        df[t_col] = ((np.isnan(padded).sum(2) == 0).cumsum(1) - 1)[is_nonempty]

        return df

    if len(padded.shape) == 2:
        padded = np.expand_dims(padded, -1)

    n_seqs, max_seq_length, n_features = padded.shape

    if ids is None:
        ids = xrange(n_seqs)

    is_nonempty = get_is_nonempty_mask(padded)

    df_new = get_basic_df(padded, is_nonempty, ids)

    for f in xrange(n_features):
        df_new = df_new.assign(tmp=padded[:, :, f][
                               is_nonempty].astype(dtypes[f]))
        df_new.rename(columns={'tmp': column_names[f]}, inplace=True)

    return df_new


def padded_events_to_tte(events, discrete_time, t_elapsed=None):
    """ computes (right censored) time to event from padded binary events.

    For details see `tte_util.get_tte`

    :param Array events: Events array.
    :param Boolean discrete_time: `True` when applying discrete time scheme.
    :param Array t_elapsed: Elapsed time. Default value is `None`.
    :return Array time_to_events: Time-to-event tensor.
    """
    seq_lengths = get_padded_seq_lengths(events)
    n_seqs = len(events)

    times_to_event = np.zeros_like(events)
    times_to_event[:] = np.nan

    t_seq = None
    for s in xrange(n_seqs):
        n = seq_lengths[s]
        if n > 0:
            event_seq = events[s, :n]
            if t_elapsed is not None:
                t_seq = t_elapsed[s, :n]

            times_to_event[s, :n] = get_tte(is_event=event_seq,
                                            discrete_time=discrete_time,
                                            t_elapsed=t_seq)

    if np.isnan(times_to_event).any():
        times_to_event[np.isnan(events)] = np.nan
    return times_to_event


def padded_events_to_not_censored_vectorized(events):
    """ (Legacy)
        calculates (non) right-censoring indicators from padded binary events
    """
    not_censored = np.zeros_like(events)
    not_censored[~np.isnan(events)] = events[~np.isnan(events)]
    # 0100 -> 0010 -> 0011 -> 1100
    not_censored = not_censored[:, ::-1].cumsum(1)[:, ::-1]

    not_censored = np.array(not_censored >= 1).astype(float)

    if np.isnan(events).any():
        not_censored[np.isnan(events)] = np.nan

    return not_censored


def padded_events_to_not_censored(events, discrete_time):
    seq_lengths = get_padded_seq_lengths(events)
    n_seqs = events.shape[0]
    is_not_censored = np.copy(events)

    for i in xrange(n_seqs):
        if seq_lengths[i] > 0:
            is_not_censored[i][:seq_lengths[i]] = get_is_not_censored(
                events[i][:seq_lengths[i]], discrete_time)
    return is_not_censored

# MISC / Data munging

# def df_to_padded_memcost(df, id_col='id', t_col='t'):
#     """
#         Calculates memory cost of padded using the alternative routes.
#         # number of arrays = features+tte+u = n_features+2
#         # To list? Pad betweeen?
#         # To array ->(pad after)
#     """

#     print('Not yet implemented')
#     return None


def _align_padded(padded, align_right):
    """ (Internal function) Aligns nan-padded temporal arrays to the right (align_right=True) or left.

    :param Array padded: padded array
    :param align_right: Determines padding orientation (right or left). If `True`, pads to right direction.
    """
    padded = np.copy(padded)

    seq_lengths = get_padded_seq_lengths(padded)
    if len(padded.shape) == 2:
        # (n_seqs,n_timesteps)
        is_flat = True
        padded = np.expand_dims(padded, -1)
    elif len(padded.shape) == 3:
        # (n_seqs,n_timesteps,n_features)
        is_flat = False
    else:
        # (n_seqs,n_timesteps,...,n_features)
        print('not yet implemented')
        # TODO

    n_seqs = padded.shape[0]
    n_timesteps = padded.shape[1]

    if align_right:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n > 0:
                padded[i, (n_timesteps - n):, :] = padded[i, :n, :]
                padded[i, :(n_timesteps - n), :] = np.nan
    else:
        for i in xrange(n_seqs):
            n = seq_lengths[i]
            if n > 0:
                padded[i, :n, :] = padded[i, (n_timesteps - n):, :]
                padded[i, n:, :] = np.nan

    if is_flat:
        padded = np.squeeze(padded)

    return padded


def right_pad_to_left_pad(padded):
    """ Change right padded to left padded. """
    return _align_padded(padded, align_right=True)


def left_pad_to_right_pad(padded):
    """ Change left padded to right padded. """
    return _align_padded(padded, align_right=False)


def df_join_in_endtime(df, constant_per_id_cols='id',
                       abs_time_col='dt',
                       abs_endtime=None,
                       fill_zeros=False):
    """ Join in NaN-rows at timestep of when we stopped observing non-events.

        If we have a dataset consisting of events recorded until a fixed
        timestamp, that timestamp won't show up in the dataset (it's a non-event).
        By joining in a row with NaN data at `abs_endtime` we get a boundarytime
        for each sequence used for TTE-calculation and padding.

        This is simpler in SQL where you join `on df.dt <= df_last_timestamp.dt`

        .. Protip::
            If discrete time: filter away last interval (ex day)
            upstream as measurements here may be incomplete, i.e if query is in
            middle of day (we are thus always looking at yesterdays data)

        :param pandas.dataframe df: Pandas dataframe
        :param constant_per_id_cols: identifying id and
                                   columns remaining constant per id&timestep
        :type constant_per_id_cols: String or String list
        :param String abs_time_col: identifying the wall-clock column df[abs_time_cols].
        :param df[abs_time_cols]) abs_endtime: The time to join in. If None it's inferred.
        :type abs_endtime: None or same as df[abs_time_cols].values.
        :param bool fill_zeros : Whether to attempt to fill NaN with zeros after merge.
        :return pandas.dataframe df: pandas dataframe where each `id` has rows at the endtime.
    """
    risky_columns = list(set(['t_elapsed', 't', 't_ix']) & set(df.columns.values))
    if len(risky_columns):
        print('Warning: df has columns ',
              risky_columns,
              ', call `df_join_in_endtime` before calculating any relative time.',
              '( otherwise they will be replaced at last step ) ')

    if type(constant_per_id_cols) is not list:
        constant_per_id_cols = [constant_per_id_cols]

    if abs_endtime is None:
        abs_endtime = df[abs_time_col].max()

    df_ids = df[constant_per_id_cols].drop_duplicates()

    df_ids[abs_time_col] = abs_endtime

    if fill_zeros:
        old_dtypes = df.dtypes.values
        cols = df.columns

        df = pd.merge(df_ids, df, how='outer').fillna(0)

        for i in xrange(len(old_dtypes)):
            df[cols[i]] = df[cols[i]].astype(old_dtypes[i])
    else:
        df = pd.merge(df_ids, df, how='outer')

    df = df.sort_values(by=[constant_per_id_cols[0], abs_time_col])

    return df


def shift_discrete_padded_features(padded, fill=0):
    """

    :param padded: padded (np array): Array [batch,timestep,...]
    :param float fill: value to replace nans with.

    =====
    Details
    =====
    For mathematical purity and to avoid confusion, in the Discrete case
    "2015-12-15" means an interval "2015-12-15 00.00 - 2015-12-15 23.59" i.e the data
    is accessible at "2015-12-15 23.59"  (time when we query our database to
    do prediction about next day.)

    In the continuous case "2015-12-15 23.59" means exactly at
    "2015-12-15 23.59: 00000000".

    Discrete case
    --------

    +-+----------------------+------+
    |t|dt                    |Event |
    +=+======================+======+
    |0|2015-12-15 00.00-23.59|1     |
    +-+----------------------+------+
    |1|2015-12-16 00.00-23.59|1     |
    +-+----------------------+------+
    |2|2015-12-17 00.00-23.59|0     |
    +-+----------------------+------+

    etc. In detail:

    +---------+-+-+-+-+-+-+----+
    |t        |0|1|2|3|4|5|....|
    +=========+=+=+=+=+=+=+====+
    |event    |1|1|0|0|1|?|....|
    +---------+-+-+-+-+-+-+----+
    |feature  |?|1|1|0|0|1|....|
    +---------+-+-+-+-+-+-+----+
    |TTE      |0|0|2|1|0|?|....|
    +---------+-+-+-+-+-+-+----+
    |Observed*|F|T|T|T|T|T|....|
    +---------+-+-+-+-+-+-+----+

    Continuous case
    --------

    +-+----------------+------+
    |t|dt              |Event |
    +=+================+======+
    |0|2015-12-15 14.39|1     |
    +-+----------------+------+
    |1|2015-12-16 16.11|1     |
    +-+----------------+------+
    |2|2015-12-17 22.18|0     |
    +-+----------------+------+

    etc. In detail:

    +---------+-+-+-+-+-+-+---+
    |t        |0|1|2|3|4|5|...|
    +=========+=+=+=+=+=+=+===+
    |event    |1|1|0|0|1|?|...|
    +---------+-+-+-+-+-+-+---+
    |feature  |1|1|0|0|1|?|...|
    +---------+-+-+-+-+-+-+---+
    |TTE      |1|3|2|1|?|?|...|
    +---------+-+-+-+-+-+-+---+
    |Observed*|T|T|T|T|T|T|...|
    +---------+-+-+-+-+-+-+---+


    *Observed = Do we have feature data at this time?*

        In the discrete case:

        -> we need to roll data intent as features to the right.

          -> First timestep typically has no measured features (and we may not even
          know until the end of the first interval if the sequence even exists!)

        So there's two options after rolling features to the right:

        1. *Fill in 0s at t=0. (`shift_discrete_padded_features`)*

            - if (data -> event) this is (randomly) leaky (potentially safe)
            - if (data <-> event) this exposes the truth (unsafe)!

        2. *Remove t=0 from target data*

            - (dont learn to predict about prospective customers first purchase)
            Safest!

        note: We never have target data for the last timestep after rolling.

      Example:
      Customer has first click leading to day 0 so at day 1 we can use
      features about that click to predict time to purchase.
      Since click does not imply purchase we can predict time to purchase
      at step 0 (but with no feature data, ex using zeros as input).
    """
    padded = np.roll(padded, shift=1, axis=1)
    padded[:, 0] = fill
    return padded

def normalize_padded(padded, means=None, stds=None):
    """Normalize by last dim of padded with means/stds or calculate them.

        .. TODO::
           * consider importing instead ex:

                from sklearn.preprocessing import StandardScaler, RobustScaler
                robust_scaler = RobustScaler()
                x_train = robust_scaler.fit_transform(x_train)
                x_test  = robust_scaler.transform(x_test)
                ValueError: Found array with dim 3. RobustScaler expected <= 2.

           * Don't normalize binary features
           * If events are sparse then this may lead to huge values.
    """
    # TODO epsilon choice is random
    epsilon = 1e-6
    original_dtype = padded.dtype

    is_flat = len(padded.shape) == 2
    if is_flat:
        padded = np.expand_dims(padded, axis=-1)

    n_features = padded.shape[2]
    n_obs = padded.shape[0] * padded.shape[1]

    if means is None:
        means = np.nanmean(np.float128(
            padded.reshape(n_obs, n_features)), axis=0)

    means = means.reshape([1, 1, n_features])
    padded = padded - means

    if stds is None:
        stds = np.nanstd(np.float128(
            padded.reshape(n_obs, n_features)), axis=0)

    stds = stds.reshape([1, 1, n_features])
    if (stds < epsilon).any():
        print('warning. Constant cols: ', np.where((stds < epsilon).flatten()))
        stds[stds < epsilon] = 1.0
        # should be (small number)/1.0 as mean is subtracted.
        # Possible prob depending on machine err

    # 128 float cast otherwise
    padded = (padded / stds).astype(original_dtype)

    if is_flat:
        # Return to flat
        padded = np.squeeze(padded)
    return padded, means, stds
