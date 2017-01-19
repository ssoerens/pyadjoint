#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Exponentiated phase misfit.

:copyright:
    created by Yanhua O. Yuan (yanhuay@princeton.edu), 2016
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from scipy.integrate import simps
from scipy import signal

from ..utils import generic_adjoint_source_plot
from ..utils import window_taper

import numpy as np

# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Exponentiated Phase Misfit"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
"""

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
# rest of the architecture of pyadjoint.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defaults to ``0.15``.

**taper_type** (:class:`str`)
    The taper type, supports anything :method:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    ret_val = {}

    measurement = []

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    adj = np.zeros(nlen_data)

    misfit_sum = 0.0

    # loop over time windows
    for wins in window:
        measure_wins = {}

        left_window_border = wins[0]
        right_window_border = wins[1]

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border - left_window_border) /
                            deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0: nlen] = observed.data[left_sample: right_sample]
        s[0: nlen] = synthetic.data[left_sample: right_sample]

        # All adjoint sources will need some kind of windowing taper
        # to get rid of kinks at two ends
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        E_s = abs(signal.hilbert(s))
        E_d = abs(signal.hilbert(d))
        Hilbt_s = np.imag(signal.hilbert(s))
        Hilbt_d = np.imag(signal.hilbert(d))

        thrd_s = config.wtr_env*E_s.max()
        thrd_d = config.wtr_env*E_d.max()
        E_s_wtr = E_s + thrd_s
        E_d_wtr = E_d + thrd_d

        diff_real = d/E_d_wtr - s/E_s_wtr
        diff_imag = Hilbt_d/E_d_wtr - Hilbt_s/E_s_wtr

        # Integrate with the composite Simpson's rule.
        misfit_real = 0.5 * simps(y=diff_real**2, dx=deltat)
        misfit_imag = 0.5 * simps(y=diff_imag**2, dx=deltat)

        misfit_sum += misfit_real + misfit_imag

        E_s_wtr_cubic = E_s_wtr**3
        adj_real = - diff_real * Hilbt_s**2 / E_s_wtr_cubic \
            - np.imag(signal.hilbert(diff_real * s * Hilbt_s / E_s_wtr_cubic))
        adj_imag = diff_imag * s * Hilbt_s / E_s_wtr_cubic \
            + np.imag(signal.hilbert(diff_imag * s**2 / E_s_wtr_cubic))

        # YY: All adjoint sources will need windowing taper again
        window_taper(adj_real, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(adj_imag, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        adj[left_sample: right_sample] = (adj_real[0:nlen] + adj_imag[0:nlen])

        measure_wins["type"] = "ep"
        measure_wins["diff_real"] = np.mean(diff_real[0:nlen])
        measure_wins["diff_imag"] = np.mean(diff_imag[0:nlen])
        measure_wins["misfit_real"] = misfit_real
        measure_wins["misfit_imag"] = misfit_imag

        measurement.append(measure_wins)

    ret_val["misfit"] = misfit_sum
    ret_val["measurement"] = measurement

    if adjoint_src is True:
        # YY: not to Reverse in time
        ret_val["adjoint_source"] = adj

    if figure:
        # return NotImplemented
        generic_adjoint_source_plot(observed, synthetic,
                                    ret_val["adjoint_source"],
                                    ret_val["misfit"],
                                    window, VERBOSE_NAME)

    return ret_val
