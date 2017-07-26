from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from six.moves import xrange

from wtte import transforms as tr


def timeline_plot(padded, title='', cmap="jet", plot=True, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))

    ax[0].imshow(padded, interpolation='none',
                 aspect='auto', cmap=cmap, origin='lower')
    ax[0].set_ylabel('sequence')
    ax[0].set_xlabel('sequence time')

    ax[1].imshow(tr.right_pad_to_left_pad(padded),
                 interpolation='none',
                 aspect='auto',
                 cmap=cmap,
                 origin='lower')
    ax[1].set_ylabel('sequence')
    ax[1].set_xlabel('absolute time')  # (Assuming sequences end today)

    fig.suptitle(title, fontsize=14)
    if plot:
        fig.show()
        return None, None
    else:
        return fig, ax


def timeline_aggregate_plot(padded, title='', cmap="jet", plot=True):
    fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True,
                           sharey=False, figsize=(12, 8))

    fig, ax[0] = timeline_plot(
        padded, title, cmap=cmap, plot=False, fig=fig, ax=ax[0])

    ax[1, 0].plot(np.nanmean(padded, axis=0), lw=0.5,
                  c='black', drawstyle='steps-post')
    ax[1, 0].set_title('mean/timestep')
    padded = tr.right_pad_to_left_pad(padded)
    ax[1, 1].plot(np.nanmean(padded, axis=0), lw=0.5,
                  c='black', drawstyle='steps-post')
    ax[1, 1].set_title('mean/timestep')

    fig.suptitle(title, fontsize=14)
    if plot:
        fig.show()
        return None, None
    else:
        return fig, ax
