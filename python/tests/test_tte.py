from __future__ import absolute_import
from __future__ import print_function
import pytest

import numpy as np
import pandas as pd
import pytest

from wtte.transforms import padded_events_to_tte, padded_events_to_not_censored

@pytest.fixture
def padded_time_continuous():
    return np.array(
        [
            [0, 1, 2, 3, 4],  # seq 1
            [0, 1, 2, 3, 4],  # seq 2..
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        ])

@pytest.fixture
def padded_time_discrete():
    return np.array(
        [
            [0, 1, 2, 3],  # seq 1
            [0, 1, 2, 3],  # seq 2..
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ])

@pytest.fixture
def events_c():
    return np.array(
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

@pytest.fixture
def expected_is_censored_c():
    return np.array(
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
@pytest.fixture
def expected_tte_c():
    return np.array(
        [
            [4, 3, 2, 1, 0],
            [4, 3, 2, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 2, 1, 0],
            [2, 1, 2, 1, 0]
        ])

@pytest.fixture
def events_d():
    return np.array(
        # Discrete events comes from continuous reality.
        [
            [0, 0, 0, 0],  # seq 1
            [0, 0, 0, 1],  # seq 2..
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ])

@pytest.fixture
def expected_is_censored_d():
    return np.array(
        [
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ])

@pytest.fixture
def expected_tte_d():
    return np.array(
        [
            [4, 3, 2, 1],
            [3, 2, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 2, 1],
            [1, 0, 2, 1]
        ])


class TestCensoringFuns:

    def test_censoring_funs_no_time_discrete(self, expected_tte_d,
                                             expected_is_censored_d, events_d):
        times_to_event = padded_events_to_tte(events_d, discrete_time=True)
        not_censored = padded_events_to_not_censored(events_d,
                                                     discrete_time=True)
 
        assert (expected_tte_d == times_to_event).all(), '  time_to_event failed'
        assert (expected_is_censored_d != not_censored).all(), 'not_censored failed'

    def test_censoring_funs_no_time_continuous(
                self, expected_tte_c, expected_is_censored_c, events_c):
        times_to_event = padded_events_to_tte(events_c, discrete_time=False)
        not_censored = padded_events_to_not_censored(events_c,
                                                     discrete_time=False)
 
        assert (expected_tte_c == times_to_event).all(), '  time_to_event failed'
        assert (expected_is_censored_c != not_censored).all(), 'not_censored failed'

    def test_censoring_funs_with_time_discrete(
                self, expected_tte_d, expected_is_censored_d, events_d,
                padded_time_discrete):
        times_to_event = padded_events_to_tte(events_d, discrete_time=True,
                                              t_elapsed=padded_time_discrete)
        not_censored = padded_events_to_not_censored(events_d,
                                                     discrete_time=True)

        assert (expected_tte_d == times_to_event).all(), '  time_to_event failed'
        assert (expected_is_censored_d != not_censored).all(), 'not_censored failed'

    def test_censoring_funs_with_time_continuous(
                self, expected_tte_c, expected_is_censored_c, events_c,
                padded_time_continuous):
        times_to_event = padded_events_to_tte(events_c, discrete_time=False,
                                              t_elapsed=padded_time_continuous)
        not_censored = padded_events_to_not_censored(events_c, discrete_time=False)

        assert (expected_tte_c == times_to_event).all(), '  time_to_event failed'
        assert (expected_is_censored_c != not_censored).all(), 'not_censored failed'

