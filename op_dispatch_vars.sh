#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
OP_SUBMISSION_VAR01_FILE="blocked_JIP_IJ.c"
OP_SUBMISSION_VAR02_FILE="blocked_JIP_IJP.c"
OP_SUBMISSION_VAR03_FILE="blocked_JIP_IP.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -mfma -fopenmp -fopt-info-vec-optimized"

