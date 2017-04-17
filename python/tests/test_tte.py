from __future__ import absolute_import
from __future__ import print_function
import pytest

import numpy as np
import pandas as pd

from wtte.transforms import padded_events_to_tte, padded_events_to_not_censored

events_c = np.array(
    # (first time is a nullity)
    # "time when something did or didn't happen"
    # Has no future importance, possibly feature importance.
    [
        [1337, 0, 0, 0, 0],  # seq 1
        [1337, 0, 0, 0, 1],  # seq 2..
        [1337, 1, 1, 1, 0],
        [1337, 1, 1, 0, 0],
        [1337, 0, 1, 0, 0]
    ])

expected_is_censored_c = np.array(
    # expected_is_censored_c[:,:-1] = expected_is_censored_d
    # Last step is always censored 
    # (observed future interval length = 0)
    [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ])

# expected_tte_c[:,:-1] = expected_tte_d+1-expected_is_censored_d
expected_tte_c = np.array(
    [
        [4, 3, 2, 1, 0],
        [4, 3, 2, 1, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 2, 1, 0],
        [2, 1, 2, 1, 0]
    ])

events_d = np.array(
# Discrete events comes from continuous reality. 
    [
        [0, 0, 0, 0],  # seq 1
        [0, 0, 0, 1],  # seq 2..
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0]
    ])

expected_is_censored_d = np.array(
    [
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    ])

expected_tte_d = np.array(
    [
        [4, 3, 2, 1],
        [3, 2, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 2, 1],
        [1, 0, 2, 1]
    ])


def test_censoring_funs():
    # TODO proper unit testing
#    print 'TTE & CENSORING'
#    print 'padded discrete'
    expected_tte = expected_tte_d
    expected_is_censored = expected_is_censored_d
    times_to_event = padded_events_to_tte(events_d, discrete_time=True)
    not_censored = padded_events_to_not_censored(
        events_d, discrete_time=True)

    assert (expected_tte == times_to_event).all(), '  time_to_event failed'
    assert (expected_is_censored != not_censored).all(), 'not_censored failed'

 #   print 'padded continuous'
    expected_tte = expected_tte_c
    expected_is_censored = expected_is_censored_c
    times_to_event = padded_events_to_tte(events_c, discrete_time=False)
    not_censored = padded_events_to_not_censored(events_c,
                                                 discrete_time=False)

    assert (expected_tte == times_to_event).all(), '  time_to_event failed'
    assert (expected_is_censored != not_censored).all(), 'not_censored failed'
