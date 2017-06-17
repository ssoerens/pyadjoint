#!/usr/bin/env pythonG
# -*- coding: utf-8 -*-
"""
Automated tests for multitaper_misfit.py and make sure it will work
and do something expected.

:copyright:
    Youyi Ruan (youyir@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import yaml
import pyadjoint
import argparse


def read_yaml_parfile(filename):
    """
    Read configure file and source type from a parameter file in ymal.
    """

    with open(filename) as fh:
        para = yaml.load(fh)

    config_dict = para["adjoint_config"]
    src_type = config_dict["adj_src_type"]

    if src_type == "multitaper_misfit":
        config = pyadjoint.ConfigMultiTaper(
            min_period=config_dict["min_period"],
            max_period=config_dict["max_period"],
            lnpt=config_dict["lnpt"],
            transfunc_waterlevel=config_dict["transfunc_waterlevel"],
            water_threshold=config_dict["water_threshold"],
            ipower_costaper=config_dict["ipower_costaper"],
            min_cycle_in_window=config_dict["min_cycle_in_window"],
            taper_percentage=config_dict["taper_percentage"],
            taper_type=config_dict["taper_type"],
            mt_nw=config_dict["mt_nw"],
            num_taper=config_dict["num_taper"],
            dt_fac=config_dict["dt_fac"],
            phase_step=config_dict["phase_step"],
            err_fac=config_dict["err_fac"],
            dt_max_scale=config_dict["dt_max_scale"],
            measure_type=config_dict["measure_type"],
            dt_sigma_min=config_dict["dt_sigma_min"],
            dlna_sigma_min=config_dict["dlna_sigma_min"],
            use_cc_error=config_dict["use_cc_error"],
            use_mt_error=config_dict["use_mt_error"])

    return config, src_type

def read_seismogram():
    # read observed and synhtetic data
    obsd, synt = pyadjoint.utils.get_example_sac_data()
    return obsd, synt

def read_window():
    # window is a list with select time windows, [start_win, end_win]
    window = [[3313.6, 3756.0]]
    return window

# multitaper_adjoint_source(adj_src):
def multitaper_adjoint_source(obsd, synt, window, config, src_type):
    """
    Example to demonstrate pyadjoint.calculate_adjoint_source
    """

    a_src = pyadjoint.calculate_adjoint_source(
        adj_src_type=src_type,
        observed=obsd,
        synthetic=synt,
        config=config,
        window=window,
        adjoint_src=True,
        plot=False)
    return a_src


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='paramfile',required=True)
    args = parser.parse_args()

    config, src_type = read_yaml_parfile(args.paramfile)
    obsd, synt = read_seismogram()
    window = read_window()

    adjsrc = multitaper_adjoint_source(obsd, synt, window, config, src_type)

    filename = "%s.%s.adj" % ("example", config.measure_type)
    adjsrc.write(filename=filename, format="SPECFEM", time_offset=0)

    for win in adjsrc.measurement:
        print ("dt: %f   misfit_dt: %f type: %s" %
               (win["dt"], win["misfit_dt"], win["type"]))

