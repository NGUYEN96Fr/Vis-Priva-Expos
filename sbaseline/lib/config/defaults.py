# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as SBASELINE

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = SBASELINE()
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1
_C.MODEL = SBASELINE()
# Setting in seed for models
_C.MODEL.SEED = 2020
# Architecture meta-data
_C.MODEL.META_ARCHITECTURE = "Learner Baseline"

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT = SBASELINE()
_C.OUTPUT.DIR = ''
_C.OUTPUT.VERBOSE = False

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = SBASELINE()
# Train, test splits of user profiles.
_C.DATASETS.TRAIN_TEST_SPLIT = ''
# Crowd-sourcing user exposure corr.
_C.DATASETS.GT_USER_EXPOS = ''
# Crowd-sourcing visual concepts in different situations
_C.DATASETS.VIS_CONCEPTS = ''

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = SBASELINE()
# Currently supported correlation types: KENDALL, PEARSON
# Evaluate the correlation score between the crowd-sourcing user exposure corr
# and the learned user exposure corr.
_C.SOLVER.CORR_TYPE = 'PEARSON'
# Cross-validation when selecting optimal subset of visual concepts
_C.SOLVER.CROSS_VAL = False
# K-fold for cross validation
_C.SOLVER.K_FOLDS = 3

# ---------------------------------------------------------------------------- #
# FOCAL EXPOSURE
# ---------------------------------------------------------------------------- #
_C.FE = SBASELINE()
# Focusing factor in the Focal Exposure (FE) function.
_C.FE.GAMMA = 0
# Scaling constant in the Focal Exposure (FE) function.
_C.FE.K = 10