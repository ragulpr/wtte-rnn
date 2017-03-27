import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO
# - Proper tests of everything
# - Leftpad instead of right-padding
# - naming in general.
# - be clearer about meaning of t_elapsed, t_ix and either (t)
# Should this be object oriented?
# Would look neat but currently been trying to keep a balance
# of having inline-able utility functions readable by data-scientists
# and translatable to other dataframe/vectorized languages.

############################## Vector based operations

def roll_fun(x, size, fun=np.mean, reverse=False):
    y = np.copy(x)
    n = len(x)
    size = min(size, n)

    if size <= 1:
        return x

    for i in xrange(size):
        y[i] = fun(x[0:(i + 1)])
    for i in xrange(size, n):
        y[i] = fun(x[(i - size + 1):(i + 1)])
    return y


def locf_if(x, is_true, reverse=False):
    """Locomote forward object x[i] if is_true[i].
        remain x untouched before first pos of truth.
    """
    if not reverse:
        for i in xrange(len(x)):
            if is_true[i]:
                cargo = x[i]
            if cargo is not None:
                x[i] = cargo
        else:
            for i in xrange(reversed(len(x))):
                if is_true[i]:
                    cargo = x[i]
                if cargo is not None:
                    x[i] = cargo
    return x


def steps_since_true_minimal(is_event):
    """(Time) since event over discrete (padded) events.
    """
    n = len(is_event)
    z = -1  # at the latest on step before
    x = np.int32(is_event)
    for i in xrange(n):
        if is_event[i]:
            z = i
        x[i] = i - z
    return x


def steps_to_true_minimal(is_event):
    """(Time) to event for discrete (padded) events.
    """
    n = len(is_event)
    z = n  # at the earliest on step after
    x = np.int32(is_event)
    for i in reversed(xrange(n)):
        if is_event[i]:
            z = i
        x[i] = z - i
    return x


def get_tte_discrete(is_event, t_elapsed=None):
    """Calculates discretely measured tte.
        Caveats:
            tte[i] = numb. timesteps to timestep with event
            Step of event has tte = 0
           (event happened at time [t,t+1))
            tte[-1]=1 if no event (censored data)
        Args:
            is_event : bolean array
            t_elapsed : int array same length as is_event.
                if None it's taken to be xrange(len(is_event))
    """
    n = len(is_event)
    tte = np.int32(is_event)
    stepsize = 1
    if t_elapsed is None:
        t_elapsed = xrange(n)

    t_next = t_elapsed[-1] + stepsize
    for i in reversed(xrange(n)):
        if is_event[i]:
            t_next = t_elapsed[i]
        tte[i] = t_next - t_elapsed[i]
    return tte


def get_tte_continuous(is_event, t_elapsed):
    """Calculates time to (pointwise measured) next event.
        Returns: diff object of time. Double, difftime, int etc.
        Caveats:
            tte[i] = time to *next* event at time t[i]
            (t[i] is exactly at event&/or query)
            tte[-1]=0 always
            (since last time is a *point*)
            Last datpoints are right censored.
        Args:
            is_event : bolean array
            t_elapsed : array same length as is_event 
                that supports vectorized subtraction
    """
    n = len(is_event)
    if t_elapsed is None:
        t_elapsed = np.int32(xrange(n))

    t_next = t_elapsed[-1]
    # lazy initialization to autoinit if difftime
    tte = t_elapsed - t_next
    for i in reversed(xrange(n)):
        tte[i] = t_next - t_elapsed[i]
        if is_event[i]:
            t_next = t_elapsed[i]
    return tte


def get_tte(is_event, is_discrete, t_elapsed=None):
    if is_discrete:
        return get_tte_discrete(is_event, t_elapsed)
    else:
        return get_tte_continuous(is_event, t_elapsed)


def get_is_not_censored(is_event, is_discrete=True):
    """ Calculates non-censoring indicator u
    """
    n = len(is_event)
    is_not_censored = np.copy(is_event)

    if is_discrete:
        # Last obs is conditionally censored
        event_seen = is_event[-1]
        for i in reversed(xrange(n)):
            if is_event[i] and not event_seen:
                event_seen = is_event[i]
            is_not_censored[i] = event_seen
    else:
        # Last obs is always censored
        event_seen = False
        for i in reversed(xrange(n)):
            is_not_censored[i] = event_seen
            if is_event[i] and not event_seen:
                event_seen = is_event[i]

    return is_not_censored

############################## Transforms

def df_to_array(df, column_names, nanpad_right=True, return_lists=False, id_col='id', t_col='t'):
    """converts flat pandas df {id,t,col1,col2,..} to array indexed [id,t,col].

    Args:
        df (pandas df): dataframe with columns 
            `id` (int), 
            `t` (int) 
            columns in `column_names` (str list)
        Where rows in df are the t'th row for a id. Think user and t'th action.
        If `t` is a non-contiguous int vec per id then steps in between t's 
        are padded with zeros. If nanpad_right sequences are np.nan-padded to max_seq_len

    Returns:
        (With seqlen the max value of `t` per id)
        `padded`:
        (if nanpad_right & !return_lists):
            a numpy float array of dimension [n_seqs,max_seqlen,n_features] 
        (if nanpad_right & return_lists):
            n_seqs numpy float sub-arrays of dimension [max_seqlen,n_features] 
        (if !nanpad_right & return_lists):
            n_seqs numpy float sub-arrays of dimension [seqlen,n_features] 
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

    Args:
        padded: a numpy float array of dimension [n_seqs,max_seqlen,n_features].
        column_names (str list): other columns to expand from df
        dtypes (str list): the type to cast the float-entries to.
        ids (list): (optional) the ids to attach to each sequence

    Returns:
        df (pandas df): dataframe with column `id` (int), and 
        `t` (int). A row in df is the t'th event for a id and has columns from column_names
    """

    def get_is_nonempty_mask(padded):
        """ 
        returns : 
            is_nonempty[i,j] : true if [i,j,:] has non-zero non-nan - entries or
            j is the start or endpoint of a sequence i
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
        """
            returns :
            df with column
             id, a column of id
             t,      a column of (user/sequence) timestep
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

############################## Awful testing

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
    # TODO proper testing
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

test_df_to_padded_padded_to_df()


def df_to_sparse_padded(df, column_names):
    # TODO
    return padded_sparse


def padded_to_timelines(padded, user_starttimes):
    """ embeds padded events on a fixed timeline
        currently only makes sense for discrete padded in between data.
        args:
            user_starttimes : datelike corresponding to entrypoint of user
        # TODO : tests and return timestamp 
    """
    seq_lengths = (False == np.isnan(padded)).sum(1)
    user_starttimes = pd.to_datetime(user_starttimes)
    timeline_start = user_starttimes.min()
    timeline_end = timeline_start + pd.DateOffset(seq_lengths.max())

    user_start_int = user_starttimes - timeline_start
    user_start_int = user_start_int.dt.components.ix[
        :, 0].values  # infer first component

    # Sort to get stepwise entry onto timeline
    m = user_start_int.argsort()
    sl_sorted = seq_lengths[m]
    user_start_int = user_start_int[m]
    padded = padded[m, :]
    user_starttimes = user_starttimes[m]

    n_timesteps = (user_start_int + sl_sorted).max().astype(int)

    n_seqs = len(user_start_int)
    padded_timelines = np.zeros([n_seqs, n_timesteps])
    padded_timelines[:, :] = np.nan

    for s in xrange(n_seqs):
        user_end_int = user_start_int[s] + sl_sorted[s]

        padded_timelines[s, user_start_int[s]:user_end_int] = padded[s, :sl_sorted[s]]
    return padded_timelines, timeline_start, timeline_end


def plot_timeline(padded_timelines, title='events'):
    # TODO dates on x-lab
    fig, ax = plt.subplots()

    ax.imshow(padded_timelines, interpolation='none',
              aspect='auto', cmap='Greys')
    ax.set_title(title)
    ax.set_ylabel('nth user')
    ax.set_xlabel('t')
    fig.gca().invert_yaxis()
    return fig, ax

############################## Calculation

def padded_events_to_tte(events, is_discrete, t_elapsed=None):
    """ computes (right censored) time to event from padded binary events.
    """
    seq_lengths = (False == np.isnan(events)).sum(1)
    n_seqs = len(events)

    times_to_event = np.zeros_like(events)
    times_to_event[:] = np.nan

    t_seq = None
    for s in xrange(n_seqs):
        n = seq_lengths[s]
        event_seq = events[s, :n]
        if t_elapsed is not None:
            t_seq = t_elapsed[s, :n]

        times_to_event[s, :n] = get_tte(is_event=event_seq,
                                        is_discrete=is_discrete,
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


def padded_events_to_not_censored(events, is_discrete):
    seq_lengths = (False == np.isnan(events)).sum(1)
    n_seqs = events.shape[0]
    is_not_censored = np.copy(events)

    for i in xrange(n_seqs):
        is_not_censored[i][:seq_lengths[i]] = get_is_not_censored(
            events[i][:seq_lengths[i]], is_discrete)
    return is_not_censored


def test_censoring_funs():
    # TODO proper unit testing
    testing_events = np.array(
        [
            [0, 0, 0, 0],  # seq 1
            [0, 0, 0, 1],  # seq 2..
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ])

    expected_tte_d = np.array(
        [
            [4, 3, 2, 1],
            [3, 2, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 2, 1],
            [1, 0, 2, 1]
        ])
    expected_tte_c = np.array(
        [
            [3, 2, 1, 0],
            [3, 2, 1, 0],
            [1, 1, 1, 0],
            [1, 2, 1, 0],
            [1, 2, 1, 0]
        ])

    expected_is_censored_d = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ])
    expected_is_censored_c = np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1]
        ])
#    print 'TTE & CENSORING'
#    print 'padded discrete'
    expected_tte = expected_tte_d
    expected_is_censored = expected_is_censored_d
    times_to_event = padded_events_to_tte(testing_events, is_discrete=True)
    not_censored = padded_events_to_not_censored(
        testing_events, is_discrete=True)

    assert (expected_tte == times_to_event).all(), '  time_to_event failed'
    assert (expected_is_censored != not_censored).all(), 'not_censored failed'

 #   print 'padded continuous'
    expected_tte = expected_tte_c
    expected_is_censored = expected_is_censored_c
    times_to_event = padded_events_to_tte(testing_events, is_discrete=False)
    not_censored = padded_events_to_not_censored(testing_events,
                                                 is_discrete=False)

    assert (expected_tte == times_to_event).all(), '  time_to_event failed'
    assert (expected_is_censored != not_censored).all(), 'not_censored failed'

test_censoring_funs()

# MISC / Data munging

def df_to_padded_memcost(df, id_col='id', t_col='t'):
    """
        Calculates memory cost of padded using the alternative routes.
        # number of arrays = features+tte+u = n_features+2
        # To list? Pad betweeen?
        # To array ->(pad after)
    """

    print('Not yet implemented')
    return None


def df_to_padded_df(df, id_col='id', t_col='t', abs_time_col='dt'):
    """zeropadds a df between timesteps.
        df with column
         id, a column of id
         t,      a column of (user/sequence) timestep
         dt, TODO expand range
         Expands each id to have to contiguous t=0,1,2..,and fills 
         NaNs with 0.
    """
    print 'warning: not tested/working'
    if abs_time_col in df.columns:
        print abs_time_col, ' filled with 0s :TODO'

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


def df_join_in_endtime(df, per_id_cols='id', abs_time_col='dt', abs_endtime=None, nanfill_val=np.nan):
    """
        Join in and fill an endtime of when we stopped observing non-events.
        TODO : Need tests

        Protip : If discrete time: filter away last interval (day) 
        upstream as measurements here may be incomplete, i.e if query is in
        middle of day (we are thus always looking at yesterdays data)
        Args:
            df : pandas datafrmae
            per_id_cols : str or list of str identifying id and static features per id 
            abs_time_col : str identifying the wall-clock column. 
            abs_endtime : type as df[abs_time_cols]). If none it's inferred.
        returns:
            df : pandas dataframe with a value 
    """
    assert 't' not in df.columns.values, 'define per-id time upstream'

    if isinstance(per_id_cols, basestring):
        per_id_cols = [per_id_cols]

    if abs_endtime is None:
        abs_endtime = df[abs_time_col].max()

    df_ids = df[per_id_cols].drop_duplicates()

    df_ids[abs_time_col] = abs_endtime

    df = pd.merge(df_ids, df, how='outer')

    df.sort_values(by=[per_id_cols[0], abs_time_col],inplace=True)
    df = df.fillna(nanfill_val)
    return df


def shift_discrete_padded_features(padded, fill=0):
    """
        Feature cols : data available at timestamp
        Target  cols : not known at timestamp

        discrete case: "event = 1 if event happens today"
         at 2015-12-15 (00:00:00) we know n_commits..
        ..to 2015-12-14 (23.59:59)
        If no event until 
            2015-12-15 (23:59:59) then event = 0
         at 2015-12-15 (23:59:59)

        continuous case: "event =1 if event happens now"
         at 2015-12-15 (00:00:00) we know n_commits..
        ..to 2015-12-15 (00:00:00)
        If no event at
            2015-12-15 (00:00:00) then event = 0
         at 2015-12-15 (00:00:00)

        -> if_discrete we need to roll data intent as features to the right. Consider this:
        As observed after the fact: 
        event   : [0,1,0,0,1]
        feature : [0,1,2,3,4]
        ...features and and target at t generated at [t,t+1)!
        As observed in realtime and what to feed to model:
        event   : [0,1,0,0,1,?]
        feature : [?,0,1,2,3,4] <- last timestep can predict but can't train
        ...features at t generated at [t-1,t), target at t generated at [t,t+1)!
          -> First timestep has no features (don't know what happened day before first day)
                 fix: set it to 0 
          -> last timestep  has no target  (don't know what can happen today)
                 fix: don't use it during training. 
        Unfortunately it usually makes sense to decide on fill-value 
        after feature normalization so do it on padded values
    """
    padded = np.roll(padded, shift=1, axis=2)
    padded[:, 0, :] = fill
    return padded


def normalize_padded(padded, means=None, stds=None):
    """ norm. by last dim of padded with norm.coef or get them.

        TODO consider importing instead ex: 
        from sklearn.preprocessing import StandardScaler, RobustScaler
        robust_scaler = RobustScaler()
        x_train = robust_scaler.fit_transform(x_train)
        x_test  = robust_scaler.transform(x_test)
        ValueError: Found array with dim 3. RobustScaler expected <= 2.

    """
    # TODO epsilon choice is random
    epsilon = 1e-8
    original_dtype = padded.dtype

    is_flat = len(padded.shape) == 2
    if is_flat:
        padded = padded.reshape([padded.shape[0], padded.shape[1], 1])

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
        print 'warning. Constant cols: ', np.where((stds < epsilon).flatten())
        stds[stds < epsilon] = 1.0
        # should be (small number)/1.0 as mean is subtracted.
        # Possible prob depending on machine err

    # 128 float cast otherwise
    padded = (padded / stds).astype(original_dtype)

    if is_flat:
        # Return to flat
        padded = np.squeeze(padded)
    return padded, means, stds

############################## Python weibull functions
def weibull_cdf(t, a, b):
    t = np.double(t)+1e-35
    return 1-np.exp(-np.power(t/a,b))

def weibull_hazard(t, a, b):
    t = np.double(t)+1e-35
    return (b/a)*np.power(t/a,b-1)

def weibull_pdf(t, a, b):
    t = np.double(t)+1e-35
    return (b/a)*np.power(t/a,b-1)*np.exp(-np.power(t/a,b))

def weibull_cmf(t, a, b):
    t = np.double(t)+1e-35
    return weibull_cdf(t+1, a, b)

def weibull_pmf(t, a, b):
    t = np.double(t)+1e-35
    return weibull_cdf(t+1.0, a, b)-weibull_cdf(t, a, b)

def weibull_mode(a, b):
    # Continuous mode.
    # TODO (mathematically) prove how close it is to discretized mode
    mode = a * np.power((b - 1.0) / b, 1.0 / b)
    mode[b <= 1.0] = 0.0
    return mode

def weibull_mean(a, b):
    # Continuous mean. at most 1 step below discretized mean 
    # E[T ] <= E[Td] + 1 true for positive distributions. 
    from scipy.special import gamma
    return a*gamma(1.0+1.0/b)

def weibull_quantiles(a, b, p):
    return a*np.power(-np.log(1.0-p),1.0/b)

def weibull_mean(a, b):
    # Continuous mean. Theoretically at most 1 step below discretized mean
    # E[T ] <= E[Td] + 1 true for positive distributions.
    from scipy.special import gamma
    return a * gamma(1.0 + 1.0 / b)

def weibull_continuous_logLik(t, a, b, u=1):
    # With equality instead of proportionality. 
    return u*np.log(weibull_pdf(t, a, b))+(1-u)*np.log(1.0-weibull_cdf(t, a, b))

def weibull_discrete_logLik(t, a, b, u=1):
    # With equality instead of proportionality. 
    return u*np.log(weibull_pmf(t, a, b))+(1-u)*np.log(1.0-weibull_cdf(t+1.0, a, b))

def weibull_cemean(t, a, b):
    # TODO this is not tested yet.
    # conditional excess mean
    # (conditional mean age at failure)
    # http://reliabilityanalyticstoolkit.appspot.com/conditional_weibull_distribution
    from scipy.special import gamma
    from scipy.special import gammainc
    # Regularized lower gamma
    print 'not tested'

    v = 1. + 1. / b
    gv = gamma(v)
    L = (t / a) ^ b
    cemean = a * gv * np.exp(L) * (1 - gammaic(v, t / a) / gv)

    return cemean

def weibull_cequantile(t, a, b, p):
    # TODO this is not tested yet.
    # conditional excess quantile
    print 'not tested'
    L = (t / a) ^ b

    quantile = a * (-np.log(1 - p) - L) ^ (1 / b)

    return quantile
