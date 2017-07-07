from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from six.moves import xrange

from .tte_util import get_tte, get_is_not_censored


def df_to_array(df, column_names, nanpad_right=True, return_lists=False, id_col='id', t_col='t'):
    """ converts flat pandas df `{id,t,col1,col2,..}` to array indexed `[id,t,col]`.

    :param df: dataframe with columns
      * `id`: integer
      * `t`: integer
      * `columns` in `column_names (String list) 
        Where rows in df are the t'th row for a id. Think user and t'th action.
        If `t` is a non-contiguous int vec per id then steps in between t's
        are padded with zeros.
    :type df: Pandas dataframe
    :param Boolean nanpad_right: If `True`,  sequences are `np.nan-padded` to `max_seq_len`
    :param return_lists:
    :param_id_col: Column where `id` is located
    :param t_col: Column where `t` is located
    :return padded: With seqlen the max value of `t` per id
      * if nanpad_right & !return_lists):
        a numpy float array of dimension `[n_seqs,max_seqlen,n_features]`
      * if nanpad_right & return_lists):
        n_seqs numpy float sub-arrays of dimension `[max_seqlen,n_features]`
      * if !nanpad_right & return_lists):
        n_seqs numpy float sub-arrays of dimension `[seqlen,n_features]`
    """

    # df.sort_values(by=['id','t'], inplace=True)
    # set the index to be this and don't drop
    df.set_index(keys=[id_col], drop=False, inplace=True)
    unique_ids = df[id_col].unique()

    n_seqs = len(unique_ids)
    n_features = len(column_names)
    seq_lengths = df[[id_col, t_col]].groupby(
        id_col).aggregate('max')[t_col].values + 1

    # We can't assume to fit varying length seqs. in flat array without
    # padding.
    assert nanpad_right or len(
        set(seq_lengths)) == 1 or return_lists, 'Wont fit in flat array'

    max_seq_len = seq_lengths.max()

    if return_lists:
        if nanpad_right:
            padded = np.split(np.zeros([n_seqs * max_seq_len, n_features]),
                              np.cumsum(np.repeat(max_seq_len, n_seqs)))
        else:
            padded = np.split(np.zeros([sum(seq_lengths), n_features]),
                              np.cumsum(seq_lengths))
    else:
        padded = np.zeros([n_seqs, max_seq_len, n_features])

    for s in xrange(n_seqs):
        # df_user is a view
        df_user = df.loc[df[id_col].values == unique_ids[s]]

        padded[s][np.array(df_user[t_col]), :] = df_user[column_names]
        if nanpad_right and seq_lengths[s] < max_seq_len:
            padded[s][seq_lengths[s]:, :].fill(np.nan)

    return padded


def df_to_padded(df, column_names, id_col='id', t_col='t'):
    """pads pandas df to a numpy array of shape [n_seqs,max_seqlen,n_features].
        see df_to_array for details
    """
    return df_to_array(df, column_names, nanpad_right=True,
                       return_lists=False, id_col=id_col, t_col=t_col)


def df_to_subarrays(df, column_names, id_col='id', t_col='t'):
    return df_to_array(df, column_names, nanpad_right=False,
                       return_lists=True, id_col=id_col, t_col=t_col)


def padded_to_df(padded, column_names, dtypes, ids=None, id_col='id', t_col='t'):
    """takes padded numpy array and converts nonzero entries to pandas dataframe row.

    Inverse to df_to_padded.
    TODO: Support mapping to non-contiguous t_col?

    :param padded: a numpy float array of dimension `[n_seqs,max_seqlen,n_features]`.
    :param column_names: other columns to expand from df
    :param dtypes:  the type to cast the float-entries to.
    :type dtypes: String list
    :param ids: (optional) the ids to attach to each sequence
    :param id_col: Column where `id` is located. Default value is `id`.
    :param t_col: Column where `t` is located. Default value is `t`.
    :return df: Dataframe with Columns
      *  `id` (Integer)
      * `t` (Integer). 
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

        # nan-mask is empty:
        is_nonempty[np.isnan(padded.sum(2))] = False

        # first and last step in each seq is not empty
        # (has info about from and to)
        is_nonempty[:, 0] = True

        is_nonempty[xrange(n_seqs), seq_lengths - 1] = True

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
        padded = padded.reshape([padded.shape[0], padded.shape[1], 1])

    n_seqs, max_seq_length, n_features = padded.shape
    seq_lengths = (np.isnan(padded).sum(2) == 0).sum(1).flatten()

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
    
    :param Array events: Events array.
    :param Boolean discrete_time: `True` when applying discrete time scheme. 
    :param Array t_elapsed: Elapsed time. Default value is `None`.
    :return Array time_to_events: Time-to-event tensor.
    """
    seq_lengths = (False == np.isnan(events)).sum(1)
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
    not_censored[np.isnan(events) == False] = events[np.isnan(events) == False]
    # 0100 -> 0010 -> 0011 -> 1100
    not_censored = not_censored[:, ::-1].cumsum(1)[:, ::-1]

    not_censored = np.array(not_censored >= 1).astype(float)

    if np.isnan(events).any():
        not_censored[np.isnan(events)] = np.nan

    return not_censored


def padded_events_to_not_censored(events, discrete_time):
    seq_lengths = (False == np.isnan(events)).sum(1)
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


def df_to_padded_df(df, id_col='id', t_col='t', abs_time_col='dt'):
    """ zeropads a df between timesteps.
        df with column
         id, a column of id
         t,      a column of (user/sequence) timestep
         dt, TODO expand range
         Expands each id to have to contiguous t=0,1,2..,and fills
         NaNs with 0.
    """
    print('warning: not tested/working')
    if abs_time_col in df.columns:
        print(abs_time_col, ' filled with 0s :TODO')

    seq_lengths = df[[id_col, t_col]].groupby(
        id_col).aggregate('max')[t_col].values + 1
    ids = np.unique(df.id.values)
    n_seqs = len(ids)

    df_new = pd.DataFrame(index=xrange(sum(seq_lengths)))

    df_new[id_col] = [ids[seq_ix]
                      for seq_ix in xrange(n_seqs) for i in xrange(seq_lengths[seq_ix])]
    df_new[t_col] = [i for seq_ix in xrange(
        n_seqs) for i in xrange(seq_lengths[seq_ix])]

    df = pd.merge(df_new, df, how='outer', on=[id_col, t_col]).fillna(0)

    return df


def _align_padded(padded, align_right):
    """ (Internal function) Aligns nan-padded temporal arrays to the right (align_right=True) or left.
    
    :param Array padded: padded array
    :param align_right: Determines padding orientation (right or left). If `True`, pads to right direction.
    """
    padded = np.copy(padded)

    if len(padded.shape) == 2:
        # (n_seqs,n_timesteps)
        seq_lengths = (False == np.isnan(padded)).sum(1)
        is_flat = True
        padded = np.expand_dims(padded, -1)
    elif len(padded.shape) == 3:
        # (n_seqs,n_timesteps,n_features,..)
        seq_lengths = (False == np.isnan(padded[:, :, 0])).sum(1)
        is_flat = False
    else:
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
                       abs_endtime=None):
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
        :param String abs_time_col: identifying the wall-clock column.
        :param df[abs_time_cols]) abs_endtime: If none it's inferred.
        :return pandas.dataframe df: pandas dataframe with a value
    """
    assert 't' not in df.columns.values, 'define elapsed time upstream'

    if type(constant_per_id_cols) is not list:
        constant_per_id_cols = [constant_per_id_cols]

    if abs_endtime is None:
        abs_endtime = df[abs_time_col].max()

    df_ids = df[constant_per_id_cols].drop_duplicates()

    df_ids[abs_time_col] = abs_endtime

    df = pd.merge(df_ids, df, how='outer')

    df = df.sort_values(by=[constant_per_id_cols[0], abs_time_col])

    return df


def shift_discrete_padded_features(padded, fill=0):
    """ Shift discrete padded features.

        Feature cols : data available at timestamp
        Target  cols : not known at timestamp

        - discrete case

            "event = 1 if event happens today"
             at 2015-12-15 (00:00:00) we know n_commits..
             to 2015-12-14 (23.59:59)
            If no event until
                2015-12-15 (23:59:59) then event = 0
             at 2015-12-15 (23:59:59)

        - continuous case

            "event =1 if event happens now"
             at 2015-12-15 (00:00:00) we know n_commits..
             to 2015-12-15 (00:00:00)
            If no event at
                2015-12-15 (00:00:00) then event = 0
             at 2015-12-15 (00:00:00)

        -> if_discrete we need to roll data intent as features to the right.

        .. Note::
            Consider this
            As observed after the fact,
                event   : [0,1,0,0,1]
                feature : [0,1,2,3,4]
            features and and target at t generated at [t,t+1)!
            
            As observed in realtime and what to feed to model
                event   : [0,1,0,0,1,?]
                feature : [?,0,1,2,3,4] <- last timestep can predict but can't train
            features at t generated at [t-1,t), target at t generated at [t,t+1)!
              -> First timestep has no features (don't know what happened day before first day)
                     fix: set it to 0
              -> last timestep  has no target  (don't know what can happen today)
                     fix: don't use it during training.
    
            Unfortunately it usually makes sense to decide on fill-value after feature normalization so do it on padded values
    """
    padded = np.roll(padded, shift=1, axis=1)
    padded[:, 0] = fill
    return padded


def normalize_padded(padded, means=None, stds=None):
    """ norm. by last dim of padded with norm.coef or get them.

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
    epsilon = 1e-8
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
        stds = np.nanmean(np.float128(
            padded.reshape(n_obs, n_features)), axis=0)

    stds = means.reshape([1, 1, n_features])
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
