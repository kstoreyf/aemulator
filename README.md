# aemulator

This repo contains scripts and notebooks used for constructing the emulators and associated routines for the paper "The Aemulus Project VI: Emulation of beyond-standard galaxy clustering statistics to improve cosmological constraints" (https://arxiv.org/abs/2210.03203).

The main scripts of interest are:
- emulator.py and run_emulator.py: Trains and tests the emulators; the emulator used in the paper is class EmulatorGeorge.
- chain.py, run_chain.py, and run_chain_mock.py: Runs the MCMC chains used in the parameter recovery tests.
- calc_covs.py: Calls routines for computing various contributions to the covariance matrices used in the analysis.

The figures in the paper were generated in this notebook: 
- 2023-06-21_aemulus_paper_figures_fmaxmocks.ipynb
