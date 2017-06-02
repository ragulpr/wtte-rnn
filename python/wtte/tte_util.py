from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

try:
    xrange
except NameError:
    xrange = range

# TODO
# - Proper tests of everything
# - naming in general.
# - be clearer about meaning of t_elapsed, t_ix and either (t)


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


def carry_forward_if(x, is_true):
    """Locomote forward object x[i] if is_true[i].
        remain x untouched before first pos of truth.
    """
    for i in xrange(len(x)):
        if is_true[i]:
            cargo = x[i]
        if cargo is not None:
            x[i] = cargo
    return x


def carry_backward_if(x, is_true):
    """Locomote backward object x[i] if is_true[i].
        remain x untouched after last pos of truth.
    """
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


def get_tte(is_event, discrete_time, t_elapsed=None):
    """(wrapper) calculates Time To Event for input vector.
    """
    if discrete_time:
        return get_tte_discrete(is_event, t_elapsed)
    else:
        return get_tte_continuous(is_event, t_elapsed)


def get_tse(is_event, t_elapsed=None):
    """(wrapper) calculates Time Since Event for input vector.

        Inverse of tte. Safe to use as a feature.
        Always "continuous" method of calculating it.
        tse >0 at time of event
            (if discrete we dont know about the event yet, if continuous
            we know at record of event so superfluous to have tse=0)
        tse = 0 at first step

        TODO reverse-indexing is pretty slow and ugly and not a
        helpful template for implementing in other languages.
    """
    if t_elapsed is not None:
        t_elapsed = t_elapsed.max() - t_elapsed[::-1]

    return get_tte_continuous(is_event[::-1], t_elapsed)[::-1]


def get_is_not_censored(is_event, discrete_time=True):
    """ Calculates non-censoring indicator u
    """
    n = len(is_event)
    is_not_censored = np.copy(is_event)

    if discrete_time:
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
