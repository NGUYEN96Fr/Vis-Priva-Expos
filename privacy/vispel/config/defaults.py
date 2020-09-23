# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import CfgNode as VISPEL

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = VISPEL()
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1
_C.MODEL = VISPEL()
# Setting in the debug mode
_C.MODEL.DEBUG = False
# Architecture meta-data
_C.MODEL.META_ARCHITECTURE = "VISual-Privacy-Exposure-Learner"

# ---------------------------------------------------------------------------- #
# OUTPUT
# ---------------------------------------------------------------------------- #
_C.OUTPUT = VISPEL()
_C.OUTPUT.DIR = ''
_C.OUTPUT.VERBOSE = False

# ---------------------------------------------------------------------------- #
# FINE TUNING
# ---------------------------------------------------------------------------- #
_C.FINE_TUNING = VISPEL()
# Fine-tune modeling parameters
_C.FINE_TUNING.STATUS = False
# Cross validation
_C.FINE_TUNING.CV = 10
# Number of used jobs
_C.FINE_TUNING.N_JOBS = -1

# -----------------------------------------------------------------------------
# DETECTOR
# -----------------------------------------------------------------------------
_C.DETECTOR = VISPEL()
# Load pre-defined detectors which was determined in
# the privacy baseline algorithm.
_C.DETECTOR.LOAD = False

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = VISPEL()
# Train, test splits of user profiles.
_C.DATASETS.TRAIN_TEST_SPLIT = ''
# Crowd-sourcing user exposure corr.
_C.DATASETS.GT_USER_EXPOS = ''
# Crowd-sourcing visual concepts in different situations
_C.DATASETS.VIS_CONCEPTS = ''
# Pre-selected visual concepts in different situations,
# given by the privacy base-line
_C.DATASETS.PRE_VIS_CONCEPTS = ''

# ---------------------------------------------------------------------------- #
# SOLVER
# ---------------------------------------------------------------------------- #
_C.SOLVER = VISPEL()
# Focusing factor in the Focal Exposure (FE) function.
_C.SOLVER.GAMMA = 2
# Scaling constant in the Focal Exposure (FE) function.
_C.SOLVER.K = 4
# Top confidence detected objects of a detector
# in a considered image.
_C.SOLVER.F_TOP = 0.3
# Select a feature type for a photo. The types include: ORG, ABS, POS_NEG
# - (ORG): Original features are the scaled positive, negative exposures
# of the photo by the FE function and the dense object score [f_expo_pos, f_expo_neg, f_dens].
# - (ABS): Sum of absolute exposure corr, and the dense object score [f_expo_pos + abs(f_expo_neg), f_dens].
# - (POS_NEG): Only the scaled positive and negative exposure corr  [f_expo_pos, f_expo_neg].
# - (SUM): Only the scaled positive and negative exposure corr  [f_expo_pos + f_expo_neg, f_dens].
_C.SOLVER.FEATURE_TYPE = 'ORG'
# Currently supported correlation types: KENDALL, PEARSON
# Evaluate the correlation score between the crowd-sourcing user exposure corr
# and the learned user exposure corr.
_C.SOLVER.CORR_TYPE = 'KENDALL'
# Filtering neutral images whose absolute exposure sum is smaller than 0.01. The
# accepted images should satisfy the following condition:
#               abs(negative_scaled_exposure) + positive_scaled_exposure > 0.01
_C.SOLVER.FILTERING = False

# ---------------------------------------------------------------------------- #
# CLUSTEROR
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR = VISPEL()
# Currently supported clustering algorithm(s):
# - k-means (K-MEANS)
# - gaussian mixture modeling (GM)
_C.CLUSTEROR.TYPE = 'K_MEANS'

# ---------------------------------------------------------------------------- #
# K MEANS
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR.K_MEANS = VISPEL()
# Number of pre-defined clusters in the K-means algorithm.
_C.CLUSTEROR.K_MEANS.CLUSTERS = 4
_C.CLUSTEROR.K_MEANS.N_INIT = 10
_C.CLUSTEROR.K_MEANS.MAX_ITER = 500
_C.CLUSTEROR.K_MEANS.ALGORITHM = 'auto'

# ---------------------------------------------------------------------------- #
# GAUSSIAN MIXTURE MODELS
# ---------------------------------------------------------------------------- #
_C.CLUSTEROR.GM = VISPEL()
_C.CLUSTEROR.GM.COMPONENTS = 4
_C.CLUSTEROR.GM.MAX_ITER = 100
_C.CLUSTEROR.GM.COV_TYPE = 'full'

# ---------------------------------------------------------------------------- #
# REGRESSOR
# ---------------------------------------------------------------------------- #
_C.REGRESSOR = VISPEL()
# Currently supported learning algorithms:
# - random forest (RF)
# - support vector machine (SVM).
_C.REGRESSOR.TYPE = 'RF'
# If use centroids (CENTROIDS) given by clustering all images in the training community
# as the features for each user's exposure. If not, the user's centroids (MEANS) calculated
# by averaging exposures on selected exposure features in each cluster will
# be taken to replace the centroids.
# Regression features, currently supported types:
# - FR1: CENTROIDS + VARIANCE ( K-MEANS, GM)
# - FR2: MEANS + VARIANCE ( K-MEANS, GM)
# - FR3: MEANS ( K-MEANS, GM)
_C.REGRESSOR.FEATURES = 'FR2'

# ---------------------------------------------------------------------------- #
# SUPPORT VECTOR MACHINE
# ---------------------------------------------------------------------------- #
_C.REGRESSOR.SVM = VISPEL()
_C.REGRESSOR.SVM.KERNEL = ['rbf']
_C.REGRESSOR.SVM.GAMMA = [1e-3]
_C.REGRESSOR.SVM.C = [5]

# ---------------------------------------------------------------------------- #
# RANDOM FOREST
# ---------------------------------------------------------------------------- #
_C.REGRESSOR.RF = VISPEL()
_C.REGRESSOR.RF.BOOTSTRAP = [True]
_C.REGRESSOR.RF.MAX_DEPTH = [7]
_C.REGRESSOR.RF.MAX_FEATURES = ['auto']
_C.REGRESSOR.RF.MIN_SAMPLES_LEAF = [1, 3, 5]
_C.REGRESSOR.RF.MIN_SAMPLES_SPLIT = [2, 4, 6]
_C.REGRESSOR.RF.N_ESTIMATORS = [150, 200]