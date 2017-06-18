#!/bin/bash
parfile="params/multitaper.adjoint.60_100.config.yml"
python example_multitaper.py \
    -p $parfile

parfile="params/cc_traveltime.adjoint.60_100.config.yml"
python example_multitaper.py \
    -p $parfile

parfile="params/waveform.adjoint.60_100.config.yml"
python example_multitaper.py \
    -p $parfile
