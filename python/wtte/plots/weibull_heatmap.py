from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from six.moves import xrange

from wtte import weibull


def basic_heatmap(ax, pred, max_horizon=None, resolution=None, cmap='jet'):
    if max_horizon is None:
        max_horizon = pred.shape[0]

    if resolution is None:
        resolution = max_horizon

    ax.imshow(pred.T, origin='lower', interpolation='none',
              aspect='auto', cmap=cmap)
    ax.set_yticks([x * (resolution + 0.0) /
                   max_horizon for x in [0, max_horizon / 2, max_horizon - 1]])
    ax.set_yticklabels([0, max_horizon / 2, max_horizon - 1])
    ax.set_ylim(-0.5, resolution - 0.5)
    ax.set_ylabel('steps ahead $s$')
    return ax


def weibull_heatmap(
    fig, ax,
    t,
    a,
    b,
    max_horizon,
    time_to_event=None,
    true_time_to_event=None,
    censoring_indicator=None,
    title='predicted Weibull pmf $p(t,s)$',
    lw=1.0,
    is_discrete=True,
    resolution=None,
    xax_nbins=10,
    yax_nbins=4,
    cmap='jet'
):
    """
        Adds a continuous or discrete heatmap with TTE to ax.

        Caveats:
        - axis are pixels so axis's always discrete.
          (so we want location of labels to be in middle)
    """
    if resolution is None:
        # Resolution. Defaults to 1/step. Want more with pdf.
        resolution = max_horizon

    # Discrete
    if is_discrete:
        prob_fun = weibull.pmf
        drawstyle = 'steps-post'
    else:
        prob_fun = weibull.pdf
        # drawstyle defaults to straight line.
        drawstyle = None

    # Number of timesteps
    n = len(t)

    # no time to event
    # no true time to event
    # no is_censored
    # all is_censored
    # ok with true_time_to_event missing but not
    # ok with true_

    if time_to_event is not None:
        if censoring_indicator is not None and true_time_to_event is None:
            is_censored = np.array(censoring_indicator).astype(bool)
        if true_time_to_event is not None:
            is_censored = (time_to_event < true_time_to_event)
        else:
            true_time_to_event = np.ones_like(time_to_event)
            true_time_to_event[:] = np.nan
            true_time_to_event[~is_censored] = time_to_event[~is_censored]

    assert len(t) == n
    assert len(a) == n
    assert len(b) == n
    assert len(time_to_event) == n
    assert len(true_time_to_event) == n
    assert len(is_censored) == n

    pred = prob_fun(
        np.tile(np.linspace(0, max_horizon - 1, resolution), (n, 1)),
        np.tile(a.reshape(n, 1), (1, resolution)),
        np.tile(b.reshape(n, 1), (1, resolution))
    )

    ax = basic_heatmap(ax, pred, max_horizon, resolution,
                       cmap=cmap)
    ax.set_title(title)

    def ax_add_scaled_line(ax, t, y, y_value_max, y_n_pixels, drawstyle,
                           linestyle='solid',
                           color='black',
                           label=None):
        # Shifts and scales y to fit on an imshow as we expect it to be, i.e
        # passing through middle of a pixel
        scaled_y = ((y_n_pixels + 0.0) / y_value_max) * y
        ax.plot(t - 0.5, scaled_y, lw=lw, linestyle=linestyle,
                drawstyle=drawstyle, color=color, label=label)
        # Adds last segment of steps-post that gets missing
        ax.plot([t[-1] - 0.5, t[-1] + 0.5], [scaled_y[-1], scaled_y[-1]],
                lw=lw,
                linestyle=linestyle,
                drawstyle=drawstyle,
                color=color)
        ax.set_xlim(-0.5, n - 0.5)

    if time_to_event is not None:
        if not all(is_censored):
            ax_add_scaled_line(ax,
                               t,
                               true_time_to_event,
                               y_value_max=max_horizon,
                               y_n_pixels=resolution,
                               drawstyle=drawstyle,
                               linestyle='solid',
                               color='black',
                               label='time_to_event')
        if not all(~is_censored):
            ax_add_scaled_line(ax,
                               t,
                               time_to_event,
                               y_value_max=max_horizon,
                               y_n_pixels=resolution,
                               drawstyle=drawstyle,
                               linestyle='dotted',
                               color='black',
                               label='(censored)')

    ax.locator_params(axis='y', nbins=4)
    ax.locator_params(axis='x', nbins=10)

#     [ax.axvline(x=k+1,lw=0.1,c='gray') for k in xrange(n-1)]

#     for k in [0,1,2]:
#         ax[k].set_xticks(ax[5].get_xticks()-0.5)
#         ax[k].set_xticklabels(ax[5].get_xticks().astype(int))

    ax.set_xlabel('time')

    fig.tight_layout()

    return fig, ax
