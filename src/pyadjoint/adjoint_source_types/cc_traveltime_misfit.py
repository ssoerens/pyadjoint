#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Cross correlation traveltime misfit.

:copyright:
    created by Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
    modified by Youyi Ruan (youyir@princeton.edu), 2016
    modified by Yanhua O. Yuan (yanhuay@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import warnings
import numpy as np
from scipy.integrate import simps

from ..utils import window_taper, generic_adjoint_source_plot
from ..config import ConfigCrossCorrelation

# FIXME: remove check over obspy version
#        Before that happen, all the processing routines in all the packages
#        should be tested and validated against the newer obspy version
from obspy import __version__ as obspy_version
if obspy_version >= u'1.0.1':
    from obspy.signal.cross_correlation import xcorr_pick_correction


VERBOSE_NAME = "Cross Correlation Traveltime Misfit"

DESCRIPTION = r"""
Traveltime misfits simply measure the squared traveltime difference. The
misfit :math:`\chi(\mathbf{m})` for a given Earth model :math:`\mathbf{m}`
and a single receiver and component is given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \left[ T^{obs} - T(\mathbf{m}) \right] ^ 2

:math:`T^{obs}` is the observed traveltime, and :math:`T(\mathbf{m})` the
predicted traveltime in Earth model :math:`\mathbf{m}`.

In practice traveltime are measured by cross correlating observed and
predicted waveforms. This particular implementation here measures cross
correlation time shifts with subsample accuracy with a fitting procedure
explained in [Deichmann1992]_. For more details see the documentation of the
:func:`~obspy.signal.cross_correlation.xcorr_pick_correction` function and the
corresponding
`Tutorial <http://docs.obspy.org/tutorial/code_snippets/xcorr_pick_correction.html>`_.


The adjoint source for the same receiver and component is then given by

.. math::

    f^{\dagger}(t) = - \left[ T^{obs} - T(\mathbf{m}) \right] ~ \frac{1}{N} ~
    \partial_t \mathbf{s}(T - t, \mathbf{m})

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


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def cc_correction(d, cc_shift, cc_dlna):
    """  correct d by shifting cc_shift and scaling exp(cc_dlna)
    """

    nlen_t = len(d)
    d_cc_dt = np.zeros(nlen_t)
    d_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - cc_shift

        if 0 <= index_shift < nlen_t:
            # corrected by c.c. shift
            d_cc_dt[index] = d[index_shift]

            # corrected by c.c. shift and amplitude
            d_cc_dtdlna[index] = np.exp(cc_dlna) * d[index_shift]

    return d_cc_dt, d_cc_dtdlna


def cc_error(d1, d2, deltat, cc_shift, cc_dlna, sigma_dt_min, sigma_dlna_min):
    """
    Estimate error for dt and dlna with uncorrelation assumption
    """

    # correct d2 by shifting cc_shift and scaling exp(cc_dlna)
    d2_cc_dt, d2_cc_dtdlna = cc_correction(d2, cc_shift, cc_dlna)

    # time derivative of d2_cc (velocity)
    d2_cc_vel = np.gradient(d2_cc_dtdlna, deltat)

    # the estimated error for dt and dlna with uncorrelation assumption
    sigma_dt_top = np.sum((d1 - d2_cc_dtdlna)**2)
    sigma_dt_bot = np.sum(d2_cc_vel**2)

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(d2_cc_dt**2)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    if sigma_dt < sigma_dt_min:
        sigma_dt = sigma_dt_min

    if sigma_dlna < sigma_dlna_min:
        sigma_dlna = sigma_dlna_min

    return sigma_dt, sigma_dlna


def cc_adj(s, cc_shift, cc_dlna, deltat, sigma_dt, sigma_dlna):
    """
    cross correlation adjoint source and misfit
    """

    nlen_t = len(s)
    fp_t = np.zeros(nlen_t)
    fq_t = np.zeros(nlen_t)
    misfit_p = 0.0
    misfit_q = 0.0

    dsdt = np.gradient(s, deltat)
    nnorm = - simps(y=dsdt*dsdt, dx=deltat)
    cc_tshift = cc_shift * deltat
    fp_t[0:nlen_t] = -1.0 * dsdt[0:nlen_t] * cc_tshift / nnorm / sigma_dt**2

    mnorm = simps(y=s*s, dx=deltat)
    fq_t[0:nlen_t] = -1.0 * s[0:nlen_t] * cc_dlna / mnorm / sigma_dlna**2

    misfit_p = 0.5 * (cc_tshift/sigma_dt)**2
    misfit_q = 0.5 * (cc_dlna/sigma_dlna)**2

    return fp_t, fq_t, misfit_p, misfit_q


def subsample_xcorr_shift(d, s):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace with subsample accuracy.
    :param s:
    :param d:
    """
    # Estimate shift and use it as a guideline for the subsample accuracy
    # shift.
    time_shift = _xcorr_shift(d.data, s.data) * d.stats.delta

    # Align on the maximum amplitude of the synthetics.
    pick_time = s.stats.starttime + s.data.argmax() * s.stats.delta

    # Will raise a warning if the trace ids don't match which we don't care
    # about here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # FIXME: remove check over obspy version
        if obspy_version >= u'1.0.1':
            return xcorr_pick_correction(
                pick_time, s, pick_time, d, 20.0 * time_shift,
                20.0 * time_shift, 10.0 * time_shift)[0]
        else:
            warnings.simplefilter("error")
            warnings.warn("Using xcorr_pick_correction requires obsy version "
                          "to be at least 1.0.1")


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    if not isinstance(config, ConfigCrossCorrelation):
        raise ValueError("Wrong configure parameters for cross correlation "
                        "adjoint source")

    ret_val_p = {}
    ret_val_q = {}

    measurement = []

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # ===
    # loop over time windows
    # ===
    for wins in window:

        measure_wins = {}

        left_window_border = wins.left
        right_window_border = wins.right

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border -
                             left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0:nlen] = observed.data[left_sample:right_sample]
        s[0:nlen] = synthetic.data[left_sample:right_sample]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        i_shift = _xcorr_shift(d, s)
        t_shift = i_shift * deltat

        cc_dlna = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) /
                               sum(s[0:nlen]*s[0:nlen]))

        # uncertainty estimate based on cross-correlations
        sigma_dt = 1.0
        sigma_dlna = 1.0

        if config.use_cc_error:
            sigma_dt, sigma_dlna = \
                    cc_error(d, s, deltat, i_shift, cc_dlna,
                             config.dt_sigma_min,
                             config.dlna_sigma_min)

        # calculate c.c. adjoint source
        fp_t, fq_t, misfit_p, misfit_q =\
            cc_adj(s, i_shift, cc_dlna, deltat,
                   sigma_dt, sigma_dlna)

        misfit_sum_p += misfit_p
        misfit_sum_q += misfit_q

        # YY: All adjoint sources will need windowing taper again
        window_taper(fp_t, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fq_t, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        fp[left_sample:right_sample] = fp_t[0:nlen]
        fq[left_sample:right_sample] = fq_t[0:nlen]

        # Taper signals following the SAC taper command
        window_taper(fp[left_sample:right_sample],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fq[left_sample:right_sample],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        measure_wins["type"] = "cc"
        measure_wins["dt"] = t_shift
        measure_wins["misfit_dt"] = misfit_p
        measure_wins["misfit_dlna"] = misfit_q

        measurement.append(measure_wins)

    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    ret_val_p["measurement"] = measurement
    ret_val_q["measurement"] = measurement

    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_p["adjoint_source"],
                                        ret_val_p["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_p

    if config.measure_type == "am":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_q["adjoint_source"],
                                        ret_val_q["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_q
