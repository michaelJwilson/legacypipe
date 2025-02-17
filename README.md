# legacypipe
Our image data reduction pipeline, using the Tractor framework

The license is 3-clause BSD.

Travis: [![Build Status](https://travis-ci.org/legacysurvey/legacypipe.svg?branch=master)](https://travis-ci.org/legacysurvey/legacypipe)
CircleCI: [![Build Status](https://img.shields.io/circleci/project/github/legacysurvey/legacypipe.svg)](https://circleci.com/gh/legacysurvey/legacypipe)
[![Docs](https://readthedocs.org/projects/legacypipe/badge/?version=latest)](http://legacypipe.readthedocs.org/en/latest/)
[![Coverage](https://coveralls.io/repos/github/legacysurvey/legacypipe/badge.svg?branch=master)](https://coveralls.io/github/legacysurvey/legacypipe)

Code for the analysis of the Legacy Surveys (DECam/DECaLS, MzLS+BASS).
========================

Some notable contents:

- legacyzpts/legacy_zeropoints.py -- code for computing photometric and astrometric zeropoints of Community Pipeline-calibrated images
- bin/runbrick-shifter.sh -- template script for our large-scale runs at NERSC
- legacypipe/runbrick.py -- the top-level script to reduce one Legacy Surveys brick.
- docker-nersc -- Dockerfile recipe for an Intel-compiler optimized build of the code, used in production
- docker -- a generic/public Dockerfile

The Docker containers we use in production are available on Docker Hub:
https://cloud.docker.com/u/legacysurvey/repository/docker/legacysurvey/legacypipe

