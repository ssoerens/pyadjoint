#!/usr/bin/env python
# -*- encoding: utf8 -*-
"""
Double-Difference Multitaper traveltime misfit.

:copyright:
    created by Yanhua O. Yuan (yanhuay@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.integrate import simps
from .. import logger
from ..utils import window_taper, generic_adjoint_source_plot
from ..config import ConfigDoubleDifferenceMultiTaper
from ..dpss import dpss_windows
from pyadjoint.adjoint_source_types.cc_traveltime_misfit \
        import _xcorr_shift, cc_error
from pyadjoint.adjoint_source_types.multitaper_misfit \
        import frequency_limit, mt_measure, mt_error
from .cc_traveltime_misfit_DD import cc_adj_DD

VERBOSE_NAME = "Double-Difference Multitaper Traveltime Misfit"

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


def mt_adj_DD(s1, s2, deltat, tapers, ddtau_mtm, ddlna_mtm, df, nlen_f,
              use_cc_error, use_mt_error, nfreq_min, nfreq_max, err_dt_cc,
              err_dlna_cc, err_dtau_mt, err_dlna_mt, wtr):

    nlen1 = len(s1)
    nlen2 = len(s2)
    if nlen1 != nlen2:
        raise ValueError("s1 and s2 are not equal in size (%d!=%d)" %
                         (nlen1, nlen2))

    nlen_t = nlen1
    ntaper = len(tapers[0])

    misfit_p = 0.0
    # misfit_q = 0.0

    # frequency-domain taper
    wp_w = np.zeros(nlen_f)
    wq_w = np.zeros(nlen_f)

    w_taper = np.zeros(nlen_f)

    win_taper_len = nfreq_max - nfreq_min
    win_taper = np.ones(win_taper_len)

    window_taper(win_taper, taper_percentage=1.0, taper_type="cos_p10")
    w_taper[nfreq_min: nfreq_max] = win_taper[0:win_taper_len]

    # normalization factor, factor 2 is needed for the integration from
    # -inf to inf
    ffac = 2.0 * df * np.sum(w_taper[nfreq_min: nfreq_max])
    logger.debug("Frequency bound (idx): [%d %d] (Hz) [%f %f]" %
                 (nfreq_min, nfreq_max-1,
                  df*(nfreq_min-1), df*(nfreq_max)))
    logger.debug("Frequency domain taper normalization coeff : %f " % ffac)
    logger.debug("Frequency domain samling length df =  %f " % df)
    if ffac <= 0.0:
        logger.warning("frequency band too narrow:")
        logger.warning("fmin=%f fmax=%f ffac=%f" %
                       (nfreq_min, nfreq_max, ffac))

    wp_w = w_taper / ffac
    wq_w = w_taper / ffac

    # cc error
    if use_cc_error:
        wp_w /= err_dt_cc**2
        wq_w /= err_dlna_cc**2

    # mt error
    if use_mt_error:
        ddtau_wtr = wtr * \
            np.sum(np.abs(ddtau_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)
        ddlna_wtr = wtr * \
            np.sum(np.abs(ddlna_mtm[nfreq_min: nfreq_max])) / \
            (nfreq_max - nfreq_min)

        err_dtau_mt[nfreq_min: nfreq_max] = \
            err_dtau_mt[nfreq_min: nfreq_max] + ddtau_wtr * \
            (err_dtau_mt[nfreq_min: nfreq_max] < ddtau_wtr)
        err_dlna_mt[nfreq_min: nfreq_max] = \
            err_dlna_mt[nfreq_min: nfreq_max] + ddlna_wtr * \
            (err_dlna_mt[nfreq_min: nfreq_max] < ddlna_wtr)

        wp_w[nfreq_min: nfreq_max] = wp_w[nfreq_min: nfreq_max] / \
            ((err_dtau_mt[nfreq_min: nfreq_max]) ** 2)
        wq_w[nfreq_min: nfreq_max] = wq_w[nfreq_min: nfreq_max] / \
            ((err_dlna_mt[nfreq_min: nfreq_max]) ** 2)

    # adjoint source
    # initialization
    bottom1 = np.zeros(nlen_f, dtype=complex)
    bottom2 = np.zeros(nlen_f, dtype=complex)
    bottom3 = np.zeros(nlen_f, dtype=complex)
    bottom4 = np.zeros(nlen_f, dtype=complex)
    bottom5 = np.zeros(nlen_f, dtype=complex)

    s1_tw = np.zeros((nlen_f, ntaper), dtype=complex)
    s1_tvw = np.zeros((nlen_f, ntaper), dtype=complex)
    s2_tw = np.zeros((nlen_f, ntaper), dtype=complex)
    s2_tvw = np.zeros((nlen_f, ntaper), dtype=complex)

    # Multitaper measurements
    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_f)
        taper[0:nlen_t] = tapers[0:nlen_t, itaper]

        # multi-tapered measurements
        s1_t = np.zeros(nlen_t)
        s1_tv = np.zeros(nlen_t)
        s2_t = np.zeros(nlen_t)
        s2_tv = np.zeros(nlen_t)

        s1_t = s1 * taper[0:nlen_t]
        s1_tv = np.gradient(s1_t, deltat)
        s2_t = s2 * taper[0:nlen_t]
        s2_tv = np.gradient(s2_t, deltat)

        # apply FFT to tapered measurements
        s1_tw[:, itaper] = np.fft.fft(s1_t, nlen_f)[:] * deltat
        s1_tvw[:, itaper] = np.fft.fft(s1_tv, nlen_f)[:] * deltat
        s2_tw[:, itaper] = np.fft.fft(s2_t, nlen_f)[:] * deltat
        s2_tvw[:, itaper] = np.fft.fft(s2_tv, nlen_f)[:] * deltat

        # calculate bottom of adjoint source
        bottom1[:] = bottom1[:] + \
            s1_tvw[:, itaper] * s2_tvw[:, itaper].conjugate()
        bottom2[:] = bottom2[:] + \
            s2_tvw[:, itaper] * s1_tvw[:, itaper].conjugate()
        bottom3[:] = bottom3[:] + \
            s1_tw[:, itaper] * s2_tw[:, itaper].conjugate()
        bottom4[:] = bottom4[:] + \
            s2_tw[:, itaper] * s1_tw[:, itaper].conjugate()
        bottom5[:] = bottom5[:] + \
            s2_tw[:, itaper] * s2_tw[:, itaper].conjugate()

    fp1_t = np.zeros(nlen_t)
    fp2_t = np.zeros(nlen_t)
    fq1_t = np.zeros(nlen_t)
    fq2_t = np.zeros(nlen_t)

    for itaper in range(0, ntaper):
        taper = np.zeros(nlen_f)
        taper[0: nlen_t] = tapers[0:nlen_t, itaper]

        # calculate pj(w), qj(w)
        p1 = np.zeros(nlen_f, dtype=complex)
        p2 = np.zeros(nlen_f, dtype=complex)
        q1 = np.zeros(nlen_f, dtype=complex)
        q2 = np.zeros(nlen_f, dtype=complex)

        p1_w = np.zeros(nlen_f, dtype=complex)
        p2_w = np.zeros(nlen_f, dtype=complex)
        q1_w = np.zeros(nlen_f, dtype=complex)
        q2_w = np.zeros(nlen_f, dtype=complex)

        p1[nfreq_min:nfreq_max] = -0.5 *\
            s2_tvw[nfreq_min:nfreq_max, itaper] / \
            (bottom2[nfreq_min:nfreq_max])
        p2[nfreq_min:nfreq_max] = 0.5 *\
            s1_tvw[nfreq_min:nfreq_max, itaper] / \
            (bottom1[nfreq_min:nfreq_max])
        q1[nfreq_min:nfreq_max] = 0.5 *\
            s2_tw[nfreq_min:nfreq_max, itaper].conjugate() / \
            (bottom3[nfreq_min:nfreq_max])
        q2[nfreq_min:nfreq_max] = 0.5 *\
            s1_tw[nfreq_min:nfreq_max, itaper].conjugate() / \
            (bottom4[nfreq_min:nfreq_max]) - \
            s2_tw[nfreq_min:nfreq_max, itaper].conjugate() / \
            (bottom5[nfreq_min:nfreq_max])

        # calculate weighted adjoint Pj(w), Qj(w)
        # adding measurement ddtau ddlna
        p1_w = 2.0 * p1 * ddtau_mtm * wp_w
        p2_w = 2.0 * p2 * ddtau_mtm * wp_w
        q1_w = (q1 + q1.conjugate()) * ddlna_mtm * wq_w
        q2_w = (q2 + q2.conjugate()) * ddlna_mtm * wq_w

        # inverse FFT to weighted adjoint (take real part)
        p1_wt = np.fft.ifft(p1_w, nlen_f).real * 2. / deltat
        p2_wt = np.fft.ifft(p2_w, nlen_f).real * 2. / deltat
        q1_wt = np.fft.ifft(q1_w, nlen_f).real * 2. / deltat
        q2_wt = np.fft.ifft(q2_w, nlen_f).real * 2. / deltat

        # apply tapering to adjoint source
        fp1_t[0:nlen_t] += p1_wt[0:nlen_t] * taper[0:nlen_t]
        fp2_t[0:nlen_t] += p2_wt[0:nlen_t] * taper[0:nlen_t]
        fq1_t[0:nlen_t] += q1_wt[0:nlen_t] * taper[0:nlen_t]
        fq2_t[0:nlen_t] += q2_wt[0:nlen_t] * taper[0:nlen_t]

    # calculate misfit
    ddtau_mtm_weigh_sqr = ddtau_mtm**2 * wp_w
    # ddlna_mtm_weigh_sqr = ddlna_mtm**2 * wq_w

    # Integrate with the composite Simpson's rule.
    misfit_p = 0.5 * 2.0 * simps(y=ddtau_mtm_weigh_sqr, dx=df)
    # misfit_q = 0.5 * 2.0 * simps(y=ddlna_mtm_weigh_sqr, dx=df)

    return fp1_t, fp2_t, misfit_p


def calculate_adjoint_source_DD(observed1, synthetic1, observed2, synthetic2,
                                config, window1, window2,
                                adjoint_src, figure):  # NOQA
    if not isinstance(config, ConfigDoubleDifferenceMultiTaper):
        raise ValueError("Wrong configure parameters for"
                         " double-difference Multitaper"
                         " adjoint source")

    # frequencies points for FFT
    nlen_f = 2**config.lnpt

    # constant for transfer function
    waterlevel_mtm = config.transfunc_waterlevel
    wtr = config.water_threshold

    # constant for cycle skip correction
    phase_step = config.phase_step

    # for frequency limit calculation
    ncycle_in_window = config.min_cycle_in_window

    # error estimation method
    use_cc_error = config.use_cc_error
    use_mt_error = config.use_mt_error

    # Frequency range for adjoint src
    min_period = config.min_period
    max_period = config.max_period

    # initialize the adjoint source
    ret_val_p1 = {}
    ret_val_p2 = {}

    # initialize the measurement dictionary
    measurement1 = []
    measurement2 = []

    nlen_data = len(synthetic1.data)
    deltat = synthetic1.stats.delta

    fp1 = np.zeros(nlen_data)
    fp2 = np.zeros(nlen_data)

    misfit_sum_p = 0.0

    # ===
    # loop over time windows in pair
    # ===
    for wins1, wins2 in zip(window1, window2):

        measure1_wins = {}
        measure2_wins = {}

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
        window_taper(d1[0:nlen1],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s1[0:nlen1],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d2[0:nlen2],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s2[0:nlen2],
                     taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        shift_obs = _xcorr_shift(d1, d2)
        cc_tshift_obs = shift_obs * deltat
        cc_dlna_obs = 0.5 *\
            np.log(sum(d1[0:nlen] * d1[0:nlen]) /
                   sum(d2[0:nlen] * d2[0:nlen]))
        shift_syn = _xcorr_shift(s1, s2)
        cc_tshift_syn = shift_syn * deltat
        cc_dlna_syn = 0.5 *\
            np.log(sum(s1[0:nlen] * s1[0:nlen]) /
                   sum(s2[0:nlen] * s2[0:nlen]))
        dd_shift = shift_syn - shift_obs
        dd_tshift = dd_shift * deltat

        # uncertainty estimate based on cross-correlations of data
        sigma_dt_cc = 1.0
        sigma_dlna_cc = 1.0

        if use_cc_error:
            sigma_dt_cc, sigma_dlna_cc = \
                    cc_error(d1, d2, deltat, shift_obs, cc_dlna_obs,
                             config.dt_sigma_min,
                             config.dlna_sigma_min)

        # re-window d1 to align with d2 for multitaper measurement
        left_sample_1d = max(left_sample_1 + shift_obs, 0)
        right_sample_1d = min(right_sample_1 + shift_obs, nlen_data)
        nlen_1d = right_sample_1d - left_sample_1d
        if nlen_1d == nlen1:
            d1_cc = np.zeros(nlen)
            # No need to correct cc_dlna in multitaper measurements
            d1_cc[0:nlen1] = observed1.data[left_sample_1d:right_sample_1d]
            d1_cc *= np.exp(-cc_dlna_obs)
            window_taper(d1_cc[0:nlen1],
                         taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
        else:
            raise Exception

        # re-window s1 to align with s2 for multitaper measurement
        left_sample_1s = max(left_sample_1 + shift_syn, 0)
        right_sample_1s = min(right_sample_1 + shift_syn, nlen_data)
        nlen_1s = right_sample_1s - left_sample_1s
        if nlen_1s == nlen1:
            s1_cc = np.zeros(nlen)
            # No need to correct cc_dlna in multitaper measurements
            s1_cc[0:nlen1] = synthetic1.data[left_sample_1s:right_sample_1s]
            s1_cc *= np.exp(-cc_dlna_syn)
            window_taper(s1_cc[0:nlen1],
                         taper_percentage=config.taper_percentage,
                         taper_type=config.taper_type)
        else:
            raise Exception

        # update
        d1 = d1_cc
        s1 = s1_cc
        right_sample_1 = right_sample_1s
        left_sample_1 = left_sample_1s

        # ===
        # Make decision wihich method to use: c.c. or multi-taper
        # always starts from multi-taper, if it doesn't work then
        # switch to cross correlation misfit
        # ===
        is_mtm = True
        is_mtm1 = True
        is_mtm2 = True

        # frequencies for FFT
        freq = np.fft.fftfreq(n=nlen_f, d=observed1.stats.delta)
        df = freq[1] - freq[0]
        wvec = freq * 2 * np.pi

        # todo: check again see if dw is not used.
        # dw = wvec[1] - wvec[0]

        # check window if okay for mtm measurements, and then find min/max
        # frequency limit for calculations.
        nfreq_min1, nfreq_max1, is_mtm1 = \
            frequency_limit(s1, nlen, nlen_f, deltat, df,
                            wtr, ncycle_in_window,
                            min_period, max_period, config.mt_nw)
        nfreq_min2, nfreq_max2, is_mtm2 = \
            frequency_limit(s2, nlen, nlen_f, deltat, df,
                            wtr, ncycle_in_window,
                            min_period, max_period, config.mt_nw)
        nfreq_min = max(nfreq_min1, nfreq_min2)
        nfreq_max = min(nfreq_max1, nfreq_max2)
        if nfreq_max <= nfreq_min or not is_mtm1 or not is_mtm2:
            is_mtm = False

        if is_mtm:
            # Set the Rayleigh bin parameter (determin taper bandwithin
            # frequency domain): nw (typical values are 2.5,3,3.5,4).
            nw = config.mt_nw
            ntaper = config.num_taper

            # generate discrete prolate slepian sequences
            tapert, eigens = dpss_windows(nlen, nw, ntaper, low_bias=False)

            if not np.isfinite(eigens).all():
                logger.warning("Error constructing dpss tapers")
                logger.warning("switch from mtm to c.c.")
                logger.debug("eigen values: %s" % eigens)
                is_mtm = False

        # check again if tapers are properly generated
        # In rare cases (e.g., [nw=2.5, nlen=61] or [nw=4.0, nlen=15]) certian
        # eigen value can not be found and associated eigen taper will be NaN
        if is_mtm:
            tapers = tapert.T

            # normalization
            tapers = tapers * np.sqrt(nlen)

            # measure frequency-dependent phase and amplitude difference
            # d1 - d2
            phi_mtm_obs = np.zeros(nlen_f)
            abs_mtm_obs = np.zeros(nlen_f)
            dtau_mtm_obs = np.zeros(nlen_f)
            dlna_mtm_obs = np.zeros(nlen_f)
            phi_mtm_obs, abs_mtm_obs, dtau_mtm_obs, dlna_mtm_obs =\
                mt_measure(d1, d2, deltat, tapers, wvec, df, nlen_f,
                           waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                           cc_tshift_obs, cc_dlna_obs)
            # s1 - s2
            phi_mtm_syn = np.zeros(nlen_f)
            abs_mtm_syn = np.zeros(nlen_f)
            dtau_mtm_syn = np.zeros(nlen_f)
            dlna_mtm_syn = np.zeros(nlen_f)
            phi_mtm_syn, abs_mtm_syn, dtau_mtm_syn, dlna_mtm_syn =\
                mt_measure(s1, s2, deltat, tapers, wvec, df, nlen_f,
                           waterlevel_mtm, phase_step, nfreq_min, nfreq_max,
                           cc_tshift_syn, cc_dlna_syn)

            # double-difference MT
            ddtau_mtm = np.zeros(nlen_f)
            ddlna_mtm = np.zeros(nlen_f)
            ddtau_mtm = dtau_mtm_syn - dtau_mtm_obs
            ddlna_mtm = dlna_mtm_syn - dlna_mtm_obs

            # multi-taper error estimation
            sigma_phi_mt = np.zeros(nlen_f)
            sigma_abs_mt = np.zeros(nlen_f)
            sigma_dtau_mt = np.zeros(nlen_f)
            sigma_dlna_mt = np.zeros(nlen_f)

            if use_mt_error:
                sigma_phi_mt, sigma_abs_mt, sigma_dtau_mt, sigma_dlna_mt =\
                    mt_error(d1, d2, deltat, tapers, wvec, df, nlen_f,
                             waterlevel_mtm, phase_step,
                             nfreq_min, nfreq_max,
                             cc_tshift_obs, cc_dlna_obs,
                             phi_mtm_obs, abs_mtm_obs,
                             dtau_mtm_obs, dlna_mtm_obs)

            # YY: check mt_measure_select

        # final decision which misfit will be used for adjoint source.
        # MT_DD
        if is_mtm:
            measure1_wins["type"] = "mt_dd_1"
            measure1_wins["ddt_w"] = ddtau_mtm[nfreq_min:nfreq_max]

            measure2_wins["type"] = "mt_dd_2"
            measure2_wins["ddt_w"] = - ddtau_mtm[nfreq_min:nfreq_max]

            # calculate multi-taper adjoint source
            fp1_t, fp2_t, misfit_p =\
                mt_adj_DD(s1, s2, deltat, tapers,
                          ddtau_mtm, ddlna_mtm, df, nlen_f,
                          use_cc_error, use_mt_error, nfreq_min, nfreq_max,
                          sigma_dt_cc, sigma_dlna_cc, sigma_dtau_mt,
                          sigma_dlna_mt, wtr)

        # CC_DD
        else:
            measure1_wins["type"] = "cc_dd_1"
            measure1_wins["ddt"] = dd_tshift

            measure2_wins["type"] = "cc_dd_2"
            measure2_wins["ddt"] = -dd_tshift

            # calculate multi-taper adjoint source
            fp1_t, fp2_t, misfit_p = \
                cc_adj_DD(s1, s2, shift_syn, dd_shift, deltat, sigma_dt_cc)

        # All adjoint sources will need windowing taper again
        window_taper(fp1_t[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(fp2_t[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        fp1[left_sample_1:right_sample_1] = fp1_t[0:nlen1]
        fp2[left_sample_2:right_sample_2] = fp2_t[0:nlen2]

        misfit_sum_p += misfit_p

        measure1_wins["misfit"] = misfit_p
        measure2_wins["misfit"] = misfit_p

        measurement1.append(measure1_wins)
        measurement2.append(measure2_wins)

    ret_val_p1["misfit"] = misfit_sum_p
    ret_val_p1["measurement"] = measurement1
    ret_val_p2["misfit"] = misfit_sum_p
    ret_val_p2["measurement"] = measurement2

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

    if config.measure_type == "dt2":
        if figure:
            generic_adjoint_source_plot(observed2, synthetic2,
                                        ret_val_p2["adjoint_source"],
                                        ret_val_p2["misfit"],
                                        window2, VERBOSE_NAME)

    return ret_val_p1, ret_val_p2
