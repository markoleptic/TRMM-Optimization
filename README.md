# TRMM Optimization

This repository implements various parallel optimization techniques for TRMM. The starting point for all variants is `baseline_op.c`.

## Optimization Techniques

- **noifstatementvarXXX.c:** Removes the if statement and reorders the loop
- **blocked_JIP_IJ_X.c:** Applies loop blocking for the I and J loops using different block sizes
- **blocked_JIP_IP_X.c:** Applies loop blocking for the I and P loops using different block sizes
- **blocked_JIP_PJ_X.c:** Applies loop blocking for the P and J loops using different block sizes
- **blocked_JIP_JIP.c:** Applies loop blocking for all loops
- **mutex_critical_section.c:** OpenMP 2 threads using `omp critical`
- **mutex_lock.c:** OpenMP 2 threads using `omp_lock_t`
- **mutex_reduction.c:** OpenMP 8 threads using `reduction(+ : res)`
- **openMP_X.c:** OpenMP using various thread sizes
- **SIMD_X.c:** Uses AVX2 to operate on eight elements at once if not along diagonal
- **openMP_SIMD.c:** Combines OpenMP and SIMD
- **tuned_variantXX_op.cu:** Applies the CUDA parallelization model to distribute work across GPU threads

## File Descriptions

- **baseline_op.c:** The starting point for all variants.
- **writeup.pdf:** Contains a full description and results of the implemented optimization techniques.

Feel free to explore each file for detailed implementation and optimization strategies.

