#!/usr/bin/env bash

######################################
# DO NOT CHANGE THIS FOLLOWING LINE: #
OP_BASELINE_FILE="baseline_op.c"    #
######################################

############################################
# HOWEVER, CHANGE THESE LINES:             #
# Replace the filenames with your variants #
############################################
# OP_SUBMISSION_VAR01_FILE="SIMD.c"
# OP_SUBMISSION_VAR02_FILE="openMP.c"
# OP_SUBMISSION_VAR03_FILE="blocked_JIP_JIP.c"
OP_SUBMISSION_VAR01_FILE="noifstatementvarIJP.c"
OP_SUBMISSION_VAR02_FILE="noifstatementvarIPJ.c"
OP_SUBMISSION_VAR03_FILE="noifstatementvarJIP.c"
OP_SUBMISSION_VAR04_FILE="noifstatementvarJPI.c"
OP_SUBMISSION_VAR05_FILE="noifstatementvarPIJ.c"
OP_SUBMISSION_VAR06_FILE="noifstatementvarPJI.c"

######################################################
# You can even change the compiler flags if you want #
######################################################
CC=mpicc
# CFLAGS="-std=c99 -O2"
CFLAGS="-std=c99 -O2 -mavx2 -mfma -fopenmp -fopt-info-vec-optimized"

