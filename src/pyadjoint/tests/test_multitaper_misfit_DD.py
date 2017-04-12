#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for DD multitaper measurement

:copyright:
    Ridvan Orsvuran (orsvuran@geoazur.unice.fr), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pytest

from pyadjoint.adjoint_source_types_DD.multitaper_misfit_DD import rewindow


@pytest.fixture
def sample_data():
    return np.arange(0, 10)


def test_rewindow_zero_shift(sample_data):
    left_sample = 1
    right_sample = 4
    shift = 0
    shifted_data, li, ri = rewindow(sample_data,
                            left_sample, right_sample, shift)
    assert np.array_equal(shifted_data, sample_data[left_sample:right_sample])


def test_rewindow_within_bounds(sample_data):
    left_sample = 1
    right_sample = 4
    shift = 2
    shifted_data, li, ri = rewindow(sample_data,
                                    left_sample, right_sample, shift)
    assert np.array_equal(shifted_data,
                          sample_data[left_sample+shift:right_sample+shift])


def test_rewindow_left_out_of_bounds(sample_data):
    left_sample = 1
    right_sample = 5
    shift = -2
    shifted_data, li, ri = rewindow(sample_data,
                                    left_sample, right_sample, shift)
    expected = np.array([0, 0, 1, 2])
    assert np.array_equal(shifted_data, expected)


def test_rewindow_right_out_of_bounds(sample_data):
    left_sample = 3
    right_sample = 8
    shift = 5
    shifted_data, li, ri = rewindow(sample_data,
                                    left_sample, right_sample, shift)
    expected = np.array([8, 9, 0, 0, 0])
    assert np.array_equal(shifted_data, expected)
