#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Double-Difference Cross correlation traveltime misfit.

:copyright:
    created by Yanhua O. Yuan (yanhuay@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.integrate import simps

from ..utils import window_taper, generic_adjoint_source_plot
from ..config import ConfigDoubleDifferenceCrossCorrelation
from pyadjoint.adjoint_source_types.cc_traveltime_misfit \
        import _xcorr_shift, cc_correction, cc_error


VERBOSE_NAME = "Double-Difference Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
Double-difference adjoint seismic tomography based on cc traveltime
Explained in [Yanhua O. Yuan; Frederik J. Simons; Jeroen Tromp
Geophys. J. Int. 2016 : ggw233v1-ggw233].
http://dx.doi.org/10.1093/gji/ggw233

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \sum_{i,j} \left[ \Delta T_{i,j}^{syn} - 
    \Delta T_{i,j}^{obs} \right] ^ 2

:math:`T_{i,j}^{syn}` is the synthetic traveltime shift of s1 and s2, 
and :math:`\Delta T_{i,j}^{obs}` the
observed traveltime shift of d1 and d2.

In practice traveltime are measured by cross correlating observed and
predicted waveforms. This particular implementation here measures cross
correlation time shifts with subsample accuracy with a fitting procedure
explained in [Deichmann1992]_. For more details see the documentation of the
:func:`~obspy.signal.cross_correlation.xcorr_pick_correction` function and the
corresponding
`Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.

We end up with a pair of adjoint source, different from conventional method using 
absolute measurement:

.. math::

    f_i^{\dagger}(t) = \sum_{j>i} \frac{\Delta \Delta t_{ij}}{N_{ij}}} 
    \partial_t s_j \big( T-[t-\Delta T_{i,j}^{syn}] \big) \\

    f_j^{\dagger}(t) = - \sum_{i<i} \frac{\Delta \Delta t_{ij}}{N_{ij}}} 
        \partial_t s_i \big( T-[t+\Delta T_{i,j}^{syn}] \big) 

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.


:math:`N` is a normalization factor given by


.. math::

    N = \int_0^T ~ \mathbf{s}(t, \mathbf{m}) ~
    \partial^2_t \mathbf{s}(t, \mathbf{m}) dt

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
"""  # NOQA

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
#  rest of the architecture.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defauls to ``0.15``.

**taper_type** (:class:`float`)
    The taper type, supports anything :meth:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


def cc_adj_DD(s1, s2, shift_syn, dd_shift, deltat, sigma_dt):
    """
    double-difference cross correlation adjoint source and misfit
    input a pair of syntheticis
    return a pair of adjoint sources
    """

    nlen1 = len(s1)
    nlen2 = len(s2)
    if nlen1 != nlen2:
        raise ValueError("s1 and s2 are not equal in size (%d!=%d)" %
                         (nlen1, nlen2))

    nlen_t = nlen1
    fp1_t = np.zeros(nlen_t)
    fp2_t = np.zeros(nlen_t)

    misfit = 0.0

    ds1_vt = np.gradient(s1, deltat)
    ds2_cc_dt, ds2_cc_dtdlna = cc_correction(s2, shift_syn, 0.0)
    ds2_cc_vt = np.gradient(ds2_cc_dt, deltat)
    ds1_cc_dt, ds1_cc_dtdlna = cc_correction(s1, -shift_syn, 0.0)
    ds1_cc_vt = np.gradient(ds1_cc_dt, deltat)
    nnorm12 = -simps(y=ds1_vt*ds2_cc_vt, dx=deltat)

    dd_tshift = dd_shift * deltat
    fp1_t = + 1.0 * ds2_cc_vt * dd_tshift / nnorm12 / sigma_dt**2
    fp2_t = - 1.0 * ds1_cc_vt * dd_shift * deltat / nnorm12 / sigma_dt**2

    misfit = 0.5 * (dd_tshift/sigma_dt)**2

    return fp1_t, fp2_t, misfit


def calculate_adjoint_source_DD(observed1, synthetic1, observed2, synthetic2,
                                config, window1, window2,
                                adjoint_src, figure):  # NOQA
    if not isinstance(config, ConfigDoubleDifferenceCrossCorrelation):
        raise ValueError("Wrong configure parameters for"
                         "double-difference cross correlation"
                         "adjoint source")

    ret_val_p1 = {}
    ret_val_p2 = {}

    measurement = []

    nlen_data = len(synthetic1.data)
    deltat = synthetic1.stats.delta

    fp1 = np.zeros(nlen_data)
    fp2 = np.zeros(nlen_data)

    misfit_sum_p = 0.0

    # ===
    # loop over time windows in pair
    # ===
    for wins1, wins2 in zip(window1, window2):

        measure_wins = {}

        left_window_border_1 = wins1[0]
        right_window_border_1 = wins1[1]
        left_window_border_2 = wins2[0]
        right_window_border_2 = wins2[1]

        left_sample_1 = int(np.floor(left_window_border_1 / deltat)) + 1
        left_sample_2 = int(np.floor(left_window_border_2 / deltat)) + 1
        nlen1 = int(np.floor((right_window_border_1 -
                             left_window_border_1) / deltat)) + 1
        nlen2 = int(np.floor((right_window_border_2 -
                             left_window_border_2) / deltat)) + 1

        right_sample_1 = left_sample_1 + nlen1
        right_sample_2 = left_sample_2 + nlen2

        nlen = max(nlen1, nlen2)

        d1 = np.zeros(nlen)
        s1 = np.zeros(nlen)
        d2 = np.zeros(nlen)
        s2 = np.zeros(nlen)

        d1[0:nlen1] = observed1.data[left_sample_1:right_sample_1]
        s1[0:nlen1] = synthetic1.data[left_sample_1:right_sample_1]
        d2[0:nlen2] = observed2.data[left_sample_2:right_sample_2]
        s2[0:nlen2] = synthetic2.data[left_sample_2:right_sample_2]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d1[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s1[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d2[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s2[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        shift_obs = _xcorr_shift(d1, d2)
        cc_dlna_obs = 0.5 * np.log(sum(d1[0:nlen]*d1[0:nlen]) /
                                   sum(d2[0:nlen]*d2[0:nlen]))
        shift_syn = _xcorr_shift(s1, s2)
        dd_shift = shift_syn - shift_obs
        dd_tshift = dd_shift * deltat

        # uncertainty estimate based on cross-correlations of data
        sigma_dt = 1.0
        sigma_dlna = 1.0

        if config.use_cc_error:
            sigma_dt, sigma_dlna = \
                    cc_error(d1, d2, deltat, shift_obs, cc_dlna_obs,
                             config.dt_sigma_min,
                             config.dlna_sigma_min)

        # calculate c.c. adjoint source
        fp1_t, fp2_t, misfit_p =\
            cc_adj_DD(s1, s2, shift_syn, dd_shift, deltat, sigma_dt)

        misfit_sum_p += misfit_p

        # YY: All adjoint sources will need windowing taper again
        window_taper(fp1_t[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fp2_t[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        fp1[left_sample_1:right_sample_1] = fp1_t[0:nlen1]
        fp2[left_sample_2:right_sample_2] = fp2_t[0:nlen2]

        measure_wins["type"] = "cc_dd"
        if config.measure_type == "dt1":
            measure_wins["dt"] = dd_tshift
        elif config.measure_type == "dt2":
            measure_wins["dt"] = - dd_tshift
        measure_wins["misfit"] = misfit_p

        measurement.append(measure_wins)

    ret_val_p1["misfit"] = misfit_sum_p
    ret_val_p1["measurement"] = measurement
    ret_val_p2["misfit"] = misfit_sum_p
    ret_val_p2["measurement"] = measurement

    if adjoint_src is True:
        # YY: not to reverse in time
        ret_val_p1["adjoint_source"] = fp1
        ret_val_p2["adjoint_source"] = fp2

    if config.measure_type == "dt1":
        if figure:
            generic_adjoint_source_plot(observed1, synthetic1,
                                        ret_val_p1["adjoint_source"],
                                        ret_val_p1["misfit"],
                                        window1, VERBOSE_NAME)

        return ret_val_p1
    if config.measure_type == "dt2":
        if figure:
            generic_adjoint_source_plot(observed2, synthetic2,
                                        ret_val_p2["adjoint_source"],
                                        ret_val_p2["misfit"],
                                        window2, VERBOSE_NAME)

        return ret_val_p2
