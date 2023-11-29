/*
  This is the baseline implementation of a Triangular Matrix Times Matrix
  Multiplication  (TRMM)

  C = AB, where
  A is an MxM lower triangular (A_{i,p} = 0 if p > i) Matrix. It is indexed by i0 and p0
  B is an MxN matrix. It is indexed by p0 and j0.
  C is an MxN matrix. It is indexed by i0 and j0.


  Parameters:

  m0 > 0: dimension
  n0 > 0: dimension



  float* A_sequential: pointer to original A matrix data
  float* A_distributed: pointer to the input data that you have distributed across
  the system

  float* C_sequential:  pointer to original output data
  float* C_distributed: pointer to the output data that you have distributed across
  the system

  float* B_sequential:  pointer to original weights data
  float* B_distributed: pointer to the weights data that you have distributed across
  the system

  Functions:

  DISTRIBUTED_ALLOCATE_NAME(...): Allocate the distributed buffers.
  DISTRIBUTE_DATA_NAME(...): takes the sequential data and distributes it across the system.
  COMPUTE_NAME(...): Performs the stencil computation.
  COLLECT_DATA_NAME(...): Collect the distributed output and combine it back to the sequential
  one for testing.
  DISTRIBUTED_FREE_NAME(...): Free the distributed buffers that were allocated


  - richard.m.veras@ou.edu

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef COMPUTE_NAME
#define COMPUTE_NAME baseline
#endif

#ifndef DISTRIBUTE_DATA_NAME
#define DISTRIBUTE_DATA_NAME baseline_distribute
#endif

#ifndef COLLECT_DATA_NAME
#define COLLECT_DATA_NAME baseline_collect
#endif

#ifndef DISTRIBUTED_ALLOCATE_NAME
#define DISTRIBUTED_ALLOCATE_NAME baseline_allocate
#endif

#ifndef DISTRIBUTED_FREE_NAME
#define DISTRIBUTED_FREE_NAME baseline_free
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void COMPUTE_NAME(int m0, int n0, float *A_distributed, float *B_distributed, float *C_distributed)

{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	/*
	  Using the convention that row_stride (rs) is the step size you take going down a row,
	  column stride (cs) is the step size going down the column.
	*/
	// A is column major
	int rs_A = m0;
	int cs_A = 1;

	// B is column major
	int rs_B = m0;
	int cs_B = 1;

	// C is column major
	int rs_C = m0;
	int cs_C = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	const int block_size = 1024;

	if (rid == root_rid)
	{
		for (int i0 = 0; i0 < n0; ++i0)
		{
			for (int p0 = 0; p0 < m0; ++p0)
			{
				C_distributed[i0 * rs_C + p0] = 0.0f;
			}
		}

		// All blocked (performs best)
		for (int j0 = 0; j0 < n0; j0 += block_size)
		{
			int jj_max = MIN(j0 + block_size, n0);
			for (int p0 = 0; p0 < m0; p0 += block_size)
			{
				int pp_max = MIN(p0 + block_size, m0);
				for (int i0 = 0; i0 <= j0; i0 += block_size)
				{
					for (int jj = j0; jj < jj_max; ++jj)
					{
						int ii_max = MIN(i0 + block_size, jj);
						for (int pp = p0; pp < pp_max; ++pp)
						{
							// if (m0 < 65 && jj == 32)
							// 	printf("jj: %d pp: %d ii: ", jj, pp);
							float B_pj = B_distributed[pp * cs_B + jj * rs_B];
							for (int ii = i0; ii < ii_max; ++ii)
							{
								// if (m0 < 65 && jj == 32)
								// 	printf("%d ", ii);
								float A_ip = A_distributed[ii * cs_A + pp * rs_A];
								// Using temp doesn't work here since it introduces
								// small floating point precision errors
								C_distributed[ii * cs_C + jj * rs_C] += A_ip * B_pj;
							}
							// if (m0 < 65 && jj == 32)
							// 	printf("\n");
						}
						// if (m0 < 65 && jj == 32)
						// 	printf("\n");
					}
				}
			}
		}

		// Blocked JI only
		// for (int j0 = 0; j0 < n0; j0 += block_size)
		// {
		// 	for (int p0 = 0; p0 < m0; ++p0)
		// 	{
		// 		for (int i0 = 0; i0 <= j0; i0 += block_size)
		// 		{
		// 			for (int jj = j0; jj < MIN(j0 + block_size, n0); ++jj)
		// 			{
		// 				float B_pj = B_distributed[p0 * cs_B + jj * rs_B];
		// 				for (int ii = i0; ii < MIN(i0 + block_size, jj); ++ii)
		// 				{
		// 					float A_ip = A_distributed[ii * cs_A + p0 * rs_A];
		// 					// Using temp doesn't work here since it introduces
		// 					// small floating point precision errors
		// 					C_distributed[ii * cs_C + jj * rs_C] += A_ip * B_pj;
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		// Blocked PI only
		// for (int j0 = 0; j0 < n0; ++j0)
		// {
		// 	for (int p0 = 0; p0 < m0; p0 += block_size)
		// 	{
		// 		for (int i0 = 0; i0 <= j0; i0 += block_size)
		// 		{
		// 			for (int pp = p0; pp < MIN(p0 + block_size, m0); ++pp)
		// 			{
		// 				float B_pj = B_distributed[pp * cs_B + j0 * rs_B];
		// 				for (int ii = i0; ii < MIN(i0 + block_size, j0); ++ii)
		// 				{
		// 					float A_ip = A_distributed[ii * cs_A + pp * rs_A];
		// 					// Using temp doesn't work here since it introduces
		// 					// small floating point precision errors
		// 					C_distributed[ii * cs_C + j0 * rs_C] += A_ip * B_pj;
		// 				}
		// 			}
		// 		}
		// 	}
		// }

		// Blocked I only
		// for (int j0 = 0; j0 < n0; ++j0)
		// {
		// 	for (int p0 = 0; p0 < m0; ++p0)
		// 	{
		// 		for (int i0 = 0; i0 <= j0; i0 += block_size)
		// 		{
		// 			float B_pj = B_distributed[p0 * cs_B + j0 * rs_B];
		// 			for (int ii = i0; ii < MIN(i0 + block_size, j0); ++ii)
		// 			{
		// 				float A_ip = A_distributed[ii * cs_A + p0 * rs_A];
		// 				// Using temp doesn't work here since it introduces
		// 				// small floating point precision errors
		// 				C_distributed[ii * cs_C + j0 * rs_C] += A_ip * B_pj;
		// 			}
		// 		}
		// 	}
		// }

		// Blocked JP only doesn't seem possible?
	}
	else
	{
		/* STUDENT_TODO: Modify this is you plan to use more
		 than 1 rank to do work in distributed memory context. */
	}
}

// Access patter for JPI JPI:
/*
jj: 0 pp: 0 ii:
jj: 0 pp: 1 ii:
jj: 0 pp: 2 ii:
jj: 0 pp: 3 ii:
jj: 0 pp: 4 ii:
jj: 0 pp: 5 ii:
jj: 0 pp: 6 ii:
jj: 0 pp: 7 ii:

jj: 1 pp: 0 ii: 0
jj: 1 pp: 1 ii: 0
jj: 1 pp: 2 ii: 0
jj: 1 pp: 3 ii: 0
jj: 1 pp: 4 ii: 0
jj: 1 pp: 5 ii: 0
jj: 1 pp: 6 ii: 0
jj: 1 pp: 7 ii: 0

jj: 2 pp: 0 ii: 0 1
jj: 2 pp: 1 ii: 0 1
jj: 2 pp: 2 ii: 0 1
jj: 2 pp: 3 ii: 0 1
jj: 2 pp: 4 ii: 0 1
jj: 2 pp: 5 ii: 0 1
jj: 2 pp: 6 ii: 0 1
jj: 2 pp: 7 ii: 0 1

jj: 3 pp: 0 ii: 0 1 2
jj: 3 pp: 1 ii: 0 1 2
jj: 3 pp: 2 ii: 0 1 2
jj: 3 pp: 3 ii: 0 1 2
jj: 3 pp: 4 ii: 0 1 2
jj: 3 pp: 5 ii: 0 1 2
jj: 3 pp: 6 ii: 0 1 2
jj: 3 pp: 7 ii: 0 1 2

jj: 4 pp: 0 ii: 0 1 2 3
jj: 4 pp: 1 ii: 0 1 2 3
jj: 4 pp: 2 ii: 0 1 2 3
jj: 4 pp: 3 ii: 0 1 2 3
jj: 4 pp: 4 ii: 0 1 2 3
jj: 4 pp: 5 ii: 0 1 2 3
jj: 4 pp: 6 ii: 0 1 2 3
jj: 4 pp: 7 ii: 0 1 2 3

jj: 5 pp: 0 ii: 0 1 2 3 4
jj: 5 pp: 1 ii: 0 1 2 3 4
jj: 5 pp: 2 ii: 0 1 2 3 4
jj: 5 pp: 3 ii: 0 1 2 3 4
jj: 5 pp: 4 ii: 0 1 2 3 4
jj: 5 pp: 5 ii: 0 1 2 3 4
jj: 5 pp: 6 ii: 0 1 2 3 4
jj: 5 pp: 7 ii: 0 1 2 3 4

jj: 6 pp: 0 ii: 0 1 2 3 4 5
jj: 6 pp: 1 ii: 0 1 2 3 4 5
jj: 6 pp: 2 ii: 0 1 2 3 4 5
jj: 6 pp: 3 ii: 0 1 2 3 4 5
jj: 6 pp: 4 ii: 0 1 2 3 4 5
jj: 6 pp: 5 ii: 0 1 2 3 4 5
jj: 6 pp: 6 ii: 0 1 2 3 4 5
jj: 6 pp: 7 ii: 0 1 2 3 4 5

jj: 7 pp: 0 ii: 0 1 2 3 4 5 6
jj: 7 pp: 1 ii: 0 1 2 3 4 5 6
jj: 7 pp: 2 ii: 0 1 2 3 4 5 6
jj: 7 pp: 3 ii: 0 1 2 3 4 5 6
jj: 7 pp: 4 ii: 0 1 2 3 4 5 6
jj: 7 pp: 5 ii: 0 1 2 3 4 5 6
jj: 7 pp: 6 ii: 0 1 2 3 4 5 6
jj: 7 pp: 7 ii: 0 1 2 3 4 5 6

jj: 0 pp: 8 ii:
jj: 0 pp: 9 ii:
jj: 0 pp: 10 ii:
jj: 0 pp: 11 ii:
jj: 0 pp: 12 ii:
jj: 0 pp: 13 ii:
jj: 0 pp: 14 ii:
jj: 0 pp: 15 ii:

jj: 1 pp: 8 ii: 0
jj: 1 pp: 9 ii: 0
jj: 1 pp: 10 ii: 0
jj: 1 pp: 11 ii: 0
jj: 1 pp: 12 ii: 0
jj: 1 pp: 13 ii: 0
jj: 1 pp: 14 ii: 0
jj: 1 pp: 15 ii: 0

jj: 2 pp: 8 ii: 0 1
jj: 2 pp: 9 ii: 0 1
jj: 2 pp: 10 ii: 0 1
jj: 2 pp: 11 ii: 0 1
jj: 2 pp: 12 ii: 0 1
jj: 2 pp: 13 ii: 0 1
jj: 2 pp: 14 ii: 0 1
jj: 2 pp: 15 ii: 0 1

jj: 3 pp: 8 ii: 0 1 2
jj: 3 pp: 9 ii: 0 1 2
jj: 3 pp: 10 ii: 0 1 2
jj: 3 pp: 11 ii: 0 1 2
jj: 3 pp: 12 ii: 0 1 2
jj: 3 pp: 13 ii: 0 1 2
jj: 3 pp: 14 ii: 0 1 2
jj: 3 pp: 15 ii: 0 1 2

jj: 4 pp: 8 ii: 0 1 2 3
jj: 4 pp: 9 ii: 0 1 2 3
jj: 4 pp: 10 ii: 0 1 2 3
jj: 4 pp: 11 ii: 0 1 2 3
jj: 4 pp: 12 ii: 0 1 2 3
jj: 4 pp: 13 ii: 0 1 2 3
jj: 4 pp: 14 ii: 0 1 2 3
jj: 4 pp: 15 ii: 0 1 2 3

jj: 5 pp: 8 ii: 0 1 2 3 4
jj: 5 pp: 9 ii: 0 1 2 3 4
jj: 5 pp: 10 ii: 0 1 2 3 4
jj: 5 pp: 11 ii: 0 1 2 3 4
jj: 5 pp: 12 ii: 0 1 2 3 4
jj: 5 pp: 13 ii: 0 1 2 3 4
jj: 5 pp: 14 ii: 0 1 2 3 4
jj: 5 pp: 15 ii: 0 1 2 3 4

jj: 6 pp: 8 ii: 0 1 2 3 4 5
jj: 6 pp: 9 ii: 0 1 2 3 4 5
jj: 6 pp: 10 ii: 0 1 2 3 4 5
jj: 6 pp: 11 ii: 0 1 2 3 4 5
jj: 6 pp: 12 ii: 0 1 2 3 4 5
jj: 6 pp: 13 ii: 0 1 2 3 4 5
jj: 6 pp: 14 ii: 0 1 2 3 4 5
jj: 6 pp: 15 ii: 0 1 2 3 4 5

jj: 7 pp: 8 ii: 0 1 2 3 4 5 6
jj: 7 pp: 9 ii: 0 1 2 3 4 5 6
jj: 7 pp: 10 ii: 0 1 2 3 4 5 6
jj: 7 pp: 11 ii: 0 1 2 3 4 5 6
jj: 7 pp: 12 ii: 0 1 2 3 4 5 6
jj: 7 pp: 13 ii: 0 1 2 3 4 5 6
jj: 7 pp: 14 ii: 0 1 2 3 4 5 6
jj: 7 pp: 15 ii: 0 1 2 3 4 5 6

jj: 0 pp: 16 ii:
jj: 0 pp: 17 ii:
jj: 0 pp: 18 ii:
jj: 0 pp: 19 ii:
jj: 0 pp: 20 ii:
jj: 0 pp: 21 ii:
jj: 0 pp: 22 ii:
jj: 0 pp: 23 ii:

jj: 1 pp: 16 ii: 0
jj: 1 pp: 17 ii: 0
jj: 1 pp: 18 ii: 0
jj: 1 pp: 19 ii: 0
jj: 1 pp: 20 ii: 0
jj: 1 pp: 21 ii: 0
jj: 1 pp: 22 ii: 0
jj: 1 pp: 23 ii: 0

jj: 2 pp: 16 ii: 0 1
jj: 2 pp: 17 ii: 0 1
jj: 2 pp: 18 ii: 0 1
jj: 2 pp: 19 ii: 0 1
jj: 2 pp: 20 ii: 0 1
jj: 2 pp: 21 ii: 0 1
jj: 2 pp: 22 ii: 0 1
jj: 2 pp: 23 ii: 0 1

jj: 3 pp: 16 ii: 0 1 2
jj: 3 pp: 17 ii: 0 1 2
jj: 3 pp: 18 ii: 0 1 2
jj: 3 pp: 19 ii: 0 1 2
jj: 3 pp: 20 ii: 0 1 2
jj: 3 pp: 21 ii: 0 1 2
jj: 3 pp: 22 ii: 0 1 2
jj: 3 pp: 23 ii: 0 1 2

jj: 4 pp: 16 ii: 0 1 2 3
jj: 4 pp: 17 ii: 0 1 2 3
jj: 4 pp: 18 ii: 0 1 2 3
jj: 4 pp: 19 ii: 0 1 2 3
jj: 4 pp: 20 ii: 0 1 2 3
jj: 4 pp: 21 ii: 0 1 2 3
jj: 4 pp: 22 ii: 0 1 2 3
jj: 4 pp: 23 ii: 0 1 2 3

jj: 5 pp: 16 ii: 0 1 2 3 4
jj: 5 pp: 17 ii: 0 1 2 3 4
jj: 5 pp: 18 ii: 0 1 2 3 4
jj: 5 pp: 19 ii: 0 1 2 3 4
jj: 5 pp: 20 ii: 0 1 2 3 4
jj: 5 pp: 21 ii: 0 1 2 3 4
jj: 5 pp: 22 ii: 0 1 2 3 4
jj: 5 pp: 23 ii: 0 1 2 3 4

jj: 6 pp: 16 ii: 0 1 2 3 4 5
jj: 6 pp: 17 ii: 0 1 2 3 4 5
jj: 6 pp: 18 ii: 0 1 2 3 4 5
jj: 6 pp: 19 ii: 0 1 2 3 4 5
jj: 6 pp: 20 ii: 0 1 2 3 4 5
jj: 6 pp: 21 ii: 0 1 2 3 4 5
jj: 6 pp: 22 ii: 0 1 2 3 4 5
jj: 6 pp: 23 ii: 0 1 2 3 4 5

jj: 7 pp: 16 ii: 0 1 2 3 4 5 6
jj: 7 pp: 17 ii: 0 1 2 3 4 5 6
jj: 7 pp: 18 ii: 0 1 2 3 4 5 6
jj: 7 pp: 19 ii: 0 1 2 3 4 5 6
jj: 7 pp: 20 ii: 0 1 2 3 4 5 6
jj: 7 pp: 21 ii: 0 1 2 3 4 5 6
jj: 7 pp: 22 ii: 0 1 2 3 4 5 6
jj: 7 pp: 23 ii: 0 1 2 3 4 5 6

jj: 0 pp: 24 ii:
jj: 0 pp: 25 ii:
jj: 0 pp: 26 ii:
jj: 0 pp: 27 ii:
jj: 0 pp: 28 ii:
jj: 0 pp: 29 ii:
jj: 0 pp: 30 ii:
jj: 0 pp: 31 ii:

jj: 1 pp: 24 ii: 0
jj: 1 pp: 25 ii: 0
jj: 1 pp: 26 ii: 0
jj: 1 pp: 27 ii: 0
jj: 1 pp: 28 ii: 0
jj: 1 pp: 29 ii: 0
jj: 1 pp: 30 ii: 0
jj: 1 pp: 31 ii: 0

jj: 2 pp: 24 ii: 0 1
jj: 2 pp: 25 ii: 0 1
jj: 2 pp: 26 ii: 0 1
jj: 2 pp: 27 ii: 0 1
jj: 2 pp: 28 ii: 0 1
jj: 2 pp: 29 ii: 0 1
jj: 2 pp: 30 ii: 0 1
jj: 2 pp: 31 ii: 0 1

jj: 3 pp: 24 ii: 0 1 2
jj: 3 pp: 25 ii: 0 1 2
jj: 3 pp: 26 ii: 0 1 2
jj: 3 pp: 27 ii: 0 1 2
jj: 3 pp: 28 ii: 0 1 2
jj: 3 pp: 29 ii: 0 1 2
jj: 3 pp: 30 ii: 0 1 2
jj: 3 pp: 31 ii: 0 1 2

jj: 4 pp: 24 ii: 0 1 2 3
jj: 4 pp: 25 ii: 0 1 2 3
jj: 4 pp: 26 ii: 0 1 2 3
jj: 4 pp: 27 ii: 0 1 2 3
jj: 4 pp: 28 ii: 0 1 2 3
jj: 4 pp: 29 ii: 0 1 2 3
jj: 4 pp: 30 ii: 0 1 2 3
jj: 4 pp: 31 ii: 0 1 2 3

jj: 5 pp: 24 ii: 0 1 2 3 4
jj: 5 pp: 25 ii: 0 1 2 3 4
jj: 5 pp: 26 ii: 0 1 2 3 4
jj: 5 pp: 27 ii: 0 1 2 3 4
jj: 5 pp: 28 ii: 0 1 2 3 4
jj: 5 pp: 29 ii: 0 1 2 3 4
jj: 5 pp: 30 ii: 0 1 2 3 4
jj: 5 pp: 31 ii: 0 1 2 3 4

jj: 6 pp: 24 ii: 0 1 2 3 4 5
jj: 6 pp: 25 ii: 0 1 2 3 4 5
jj: 6 pp: 26 ii: 0 1 2 3 4 5
jj: 6 pp: 27 ii: 0 1 2 3 4 5
jj: 6 pp: 28 ii: 0 1 2 3 4 5
jj: 6 pp: 29 ii: 0 1 2 3 4 5
jj: 6 pp: 30 ii: 0 1 2 3 4 5
jj: 6 pp: 31 ii: 0 1 2 3 4 5

jj: 7 pp: 24 ii: 0 1 2 3 4 5 6
jj: 7 pp: 25 ii: 0 1 2 3 4 5 6
jj: 7 pp: 26 ii: 0 1 2 3 4 5 6
jj: 7 pp: 27 ii: 0 1 2 3 4 5 6
jj: 7 pp: 28 ii: 0 1 2 3 4 5 6
jj: 7 pp: 29 ii: 0 1 2 3 4 5 6
jj: 7 pp: 30 ii: 0 1 2 3 4 5 6
jj: 7 pp: 31 ii: 0 1 2 3 4 5 6

jj: 0 pp: 32 ii:
jj: 0 pp: 33 ii:
jj: 0 pp: 34 ii:
jj: 0 pp: 35 ii:
jj: 0 pp: 36 ii:
jj: 0 pp: 37 ii:
jj: 0 pp: 38 ii:
jj: 0 pp: 39 ii:

jj: 1 pp: 32 ii: 0
jj: 1 pp: 33 ii: 0
jj: 1 pp: 34 ii: 0
jj: 1 pp: 35 ii: 0
jj: 1 pp: 36 ii: 0
jj: 1 pp: 37 ii: 0
jj: 1 pp: 38 ii: 0
jj: 1 pp: 39 ii: 0

jj: 2 pp: 32 ii: 0 1
jj: 2 pp: 33 ii: 0 1
jj: 2 pp: 34 ii: 0 1
jj: 2 pp: 35 ii: 0 1
jj: 2 pp: 36 ii: 0 1
jj: 2 pp: 37 ii: 0 1
jj: 2 pp: 38 ii: 0 1
jj: 2 pp: 39 ii: 0 1

jj: 3 pp: 32 ii: 0 1 2
jj: 3 pp: 33 ii: 0 1 2
jj: 3 pp: 34 ii: 0 1 2
jj: 3 pp: 35 ii: 0 1 2
jj: 3 pp: 36 ii: 0 1 2
jj: 3 pp: 37 ii: 0 1 2
jj: 3 pp: 38 ii: 0 1 2
jj: 3 pp: 39 ii: 0 1 2

jj: 4 pp: 32 ii: 0 1 2 3
jj: 4 pp: 33 ii: 0 1 2 3
jj: 4 pp: 34 ii: 0 1 2 3
jj: 4 pp: 35 ii: 0 1 2 3
jj: 4 pp: 36 ii: 0 1 2 3
jj: 4 pp: 37 ii: 0 1 2 3
jj: 4 pp: 38 ii: 0 1 2 3
jj: 4 pp: 39 ii: 0 1 2 3

jj: 5 pp: 32 ii: 0 1 2 3 4
jj: 5 pp: 33 ii: 0 1 2 3 4
jj: 5 pp: 34 ii: 0 1 2 3 4
jj: 5 pp: 35 ii: 0 1 2 3 4
jj: 5 pp: 36 ii: 0 1 2 3 4
jj: 5 pp: 37 ii: 0 1 2 3 4
jj: 5 pp: 38 ii: 0 1 2 3 4
jj: 5 pp: 39 ii: 0 1 2 3 4

jj: 6 pp: 32 ii: 0 1 2 3 4 5
jj: 6 pp: 33 ii: 0 1 2 3 4 5
jj: 6 pp: 34 ii: 0 1 2 3 4 5
jj: 6 pp: 35 ii: 0 1 2 3 4 5
jj: 6 pp: 36 ii: 0 1 2 3 4 5
jj: 6 pp: 37 ii: 0 1 2 3 4 5
jj: 6 pp: 38 ii: 0 1 2 3 4 5
jj: 6 pp: 39 ii: 0 1 2 3 4 5

jj: 7 pp: 32 ii: 0 1 2 3 4 5 6
jj: 7 pp: 33 ii: 0 1 2 3 4 5 6
jj: 7 pp: 34 ii: 0 1 2 3 4 5 6
jj: 7 pp: 35 ii: 0 1 2 3 4 5 6
jj: 7 pp: 36 ii: 0 1 2 3 4 5 6
jj: 7 pp: 37 ii: 0 1 2 3 4 5 6
jj: 7 pp: 38 ii: 0 1 2 3 4 5 6
jj: 7 pp: 39 ii: 0 1 2 3 4 5 6

jj: 0 pp: 40 ii:
jj: 0 pp: 41 ii:
jj: 0 pp: 42 ii:
jj: 0 pp: 43 ii:
jj: 0 pp: 44 ii:
jj: 0 pp: 45 ii:
jj: 0 pp: 46 ii:
jj: 0 pp: 47 ii:

jj: 1 pp: 40 ii: 0
jj: 1 pp: 41 ii: 0
jj: 1 pp: 42 ii: 0
jj: 1 pp: 43 ii: 0
jj: 1 pp: 44 ii: 0
jj: 1 pp: 45 ii: 0
jj: 1 pp: 46 ii: 0
jj: 1 pp: 47 ii: 0

jj: 2 pp: 40 ii: 0 1
jj: 2 pp: 41 ii: 0 1
jj: 2 pp: 42 ii: 0 1
jj: 2 pp: 43 ii: 0 1
jj: 2 pp: 44 ii: 0 1
jj: 2 pp: 45 ii: 0 1
jj: 2 pp: 46 ii: 0 1
jj: 2 pp: 47 ii: 0 1

jj: 3 pp: 40 ii: 0 1 2
jj: 3 pp: 41 ii: 0 1 2
jj: 3 pp: 42 ii: 0 1 2
jj: 3 pp: 43 ii: 0 1 2
jj: 3 pp: 44 ii: 0 1 2
jj: 3 pp: 45 ii: 0 1 2
jj: 3 pp: 46 ii: 0 1 2
jj: 3 pp: 47 ii: 0 1 2

jj: 4 pp: 40 ii: 0 1 2 3
jj: 4 pp: 41 ii: 0 1 2 3
jj: 4 pp: 42 ii: 0 1 2 3
jj: 4 pp: 43 ii: 0 1 2 3
jj: 4 pp: 44 ii: 0 1 2 3
jj: 4 pp: 45 ii: 0 1 2 3
jj: 4 pp: 46 ii: 0 1 2 3
jj: 4 pp: 47 ii: 0 1 2 3

jj: 5 pp: 40 ii: 0 1 2 3 4
jj: 5 pp: 41 ii: 0 1 2 3 4
jj: 5 pp: 42 ii: 0 1 2 3 4
jj: 5 pp: 43 ii: 0 1 2 3 4
jj: 5 pp: 44 ii: 0 1 2 3 4
jj: 5 pp: 45 ii: 0 1 2 3 4
jj: 5 pp: 46 ii: 0 1 2 3 4
jj: 5 pp: 47 ii: 0 1 2 3 4

jj: 6 pp: 40 ii: 0 1 2 3 4 5
jj: 6 pp: 41 ii: 0 1 2 3 4 5
jj: 6 pp: 42 ii: 0 1 2 3 4 5
jj: 6 pp: 43 ii: 0 1 2 3 4 5
jj: 6 pp: 44 ii: 0 1 2 3 4 5
jj: 6 pp: 45 ii: 0 1 2 3 4 5
jj: 6 pp: 46 ii: 0 1 2 3 4 5
jj: 6 pp: 47 ii: 0 1 2 3 4 5

jj: 7 pp: 40 ii: 0 1 2 3 4 5 6
jj: 7 pp: 41 ii: 0 1 2 3 4 5 6
jj: 7 pp: 42 ii: 0 1 2 3 4 5 6
jj: 7 pp: 43 ii: 0 1 2 3 4 5 6
jj: 7 pp: 44 ii: 0 1 2 3 4 5 6
jj: 7 pp: 45 ii: 0 1 2 3 4 5 6
jj: 7 pp: 46 ii: 0 1 2 3 4 5 6
jj: 7 pp: 47 ii: 0 1 2 3 4 5 6

jj: 0 pp: 48 ii:
jj: 0 pp: 49 ii:
jj: 0 pp: 50 ii:
jj: 0 pp: 51 ii:
jj: 0 pp: 52 ii:
jj: 0 pp: 53 ii:
jj: 0 pp: 54 ii:
jj: 0 pp: 55 ii:

jj: 1 pp: 48 ii: 0
jj: 1 pp: 49 ii: 0
jj: 1 pp: 50 ii: 0
jj: 1 pp: 51 ii: 0
jj: 1 pp: 52 ii: 0
jj: 1 pp: 53 ii: 0
jj: 1 pp: 54 ii: 0
jj: 1 pp: 55 ii: 0

jj: 2 pp: 48 ii: 0 1
jj: 2 pp: 49 ii: 0 1
jj: 2 pp: 50 ii: 0 1
jj: 2 pp: 51 ii: 0 1
jj: 2 pp: 52 ii: 0 1
jj: 2 pp: 53 ii: 0 1
jj: 2 pp: 54 ii: 0 1
jj: 2 pp: 55 ii: 0 1

jj: 3 pp: 48 ii: 0 1 2
jj: 3 pp: 49 ii: 0 1 2
jj: 3 pp: 50 ii: 0 1 2
jj: 3 pp: 51 ii: 0 1 2
jj: 3 pp: 52 ii: 0 1 2
jj: 3 pp: 53 ii: 0 1 2
jj: 3 pp: 54 ii: 0 1 2
jj: 3 pp: 55 ii: 0 1 2

jj: 4 pp: 48 ii: 0 1 2 3
jj: 4 pp: 49 ii: 0 1 2 3
jj: 4 pp: 50 ii: 0 1 2 3
jj: 4 pp: 51 ii: 0 1 2 3
jj: 4 pp: 52 ii: 0 1 2 3
jj: 4 pp: 53 ii: 0 1 2 3
jj: 4 pp: 54 ii: 0 1 2 3
jj: 4 pp: 55 ii: 0 1 2 3

jj: 5 pp: 48 ii: 0 1 2 3 4
jj: 5 pp: 49 ii: 0 1 2 3 4
jj: 5 pp: 50 ii: 0 1 2 3 4
jj: 5 pp: 51 ii: 0 1 2 3 4
jj: 5 pp: 52 ii: 0 1 2 3 4
jj: 5 pp: 53 ii: 0 1 2 3 4
jj: 5 pp: 54 ii: 0 1 2 3 4
jj: 5 pp: 55 ii: 0 1 2 3 4

jj: 6 pp: 48 ii: 0 1 2 3 4 5
jj: 6 pp: 49 ii: 0 1 2 3 4 5
jj: 6 pp: 50 ii: 0 1 2 3 4 5
jj: 6 pp: 51 ii: 0 1 2 3 4 5
jj: 6 pp: 52 ii: 0 1 2 3 4 5
jj: 6 pp: 53 ii: 0 1 2 3 4 5
jj: 6 pp: 54 ii: 0 1 2 3 4 5
jj: 6 pp: 55 ii: 0 1 2 3 4 5

jj: 7 pp: 48 ii: 0 1 2 3 4 5 6
jj: 7 pp: 49 ii: 0 1 2 3 4 5 6
jj: 7 pp: 50 ii: 0 1 2 3 4 5 6
jj: 7 pp: 51 ii: 0 1 2 3 4 5 6
jj: 7 pp: 52 ii: 0 1 2 3 4 5 6
jj: 7 pp: 53 ii: 0 1 2 3 4 5 6
jj: 7 pp: 54 ii: 0 1 2 3 4 5 6
jj: 7 pp: 55 ii: 0 1 2 3 4 5 6

jj: 0 pp: 56 ii:
jj: 0 pp: 57 ii:
jj: 0 pp: 58 ii:
jj: 0 pp: 59 ii:
jj: 0 pp: 60 ii:
jj: 0 pp: 61 ii:
jj: 0 pp: 62 ii:
jj: 0 pp: 63 ii:

jj: 1 pp: 56 ii: 0
jj: 1 pp: 57 ii: 0
jj: 1 pp: 58 ii: 0
jj: 1 pp: 59 ii: 0
jj: 1 pp: 60 ii: 0
jj: 1 pp: 61 ii: 0
jj: 1 pp: 62 ii: 0
jj: 1 pp: 63 ii: 0

jj: 2 pp: 56 ii: 0 1
jj: 2 pp: 57 ii: 0 1
jj: 2 pp: 58 ii: 0 1
jj: 2 pp: 59 ii: 0 1
jj: 2 pp: 60 ii: 0 1
jj: 2 pp: 61 ii: 0 1
jj: 2 pp: 62 ii: 0 1
jj: 2 pp: 63 ii: 0 1

jj: 3 pp: 56 ii: 0 1 2
jj: 3 pp: 57 ii: 0 1 2
jj: 3 pp: 58 ii: 0 1 2
jj: 3 pp: 59 ii: 0 1 2
jj: 3 pp: 60 ii: 0 1 2
jj: 3 pp: 61 ii: 0 1 2
jj: 3 pp: 62 ii: 0 1 2
jj: 3 pp: 63 ii: 0 1 2

jj: 4 pp: 56 ii: 0 1 2 3
jj: 4 pp: 57 ii: 0 1 2 3
jj: 4 pp: 58 ii: 0 1 2 3
jj: 4 pp: 59 ii: 0 1 2 3
jj: 4 pp: 60 ii: 0 1 2 3
jj: 4 pp: 61 ii: 0 1 2 3
jj: 4 pp: 62 ii: 0 1 2 3
jj: 4 pp: 63 ii: 0 1 2 3

jj: 5 pp: 56 ii: 0 1 2 3 4
jj: 5 pp: 57 ii: 0 1 2 3 4
jj: 5 pp: 58 ii: 0 1 2 3 4
jj: 5 pp: 59 ii: 0 1 2 3 4
jj: 5 pp: 60 ii: 0 1 2 3 4
jj: 5 pp: 61 ii: 0 1 2 3 4
jj: 5 pp: 62 ii: 0 1 2 3 4
jj: 5 pp: 63 ii: 0 1 2 3 4

jj: 6 pp: 56 ii: 0 1 2 3 4 5
jj: 6 pp: 57 ii: 0 1 2 3 4 5
jj: 6 pp: 58 ii: 0 1 2 3 4 5
jj: 6 pp: 59 ii: 0 1 2 3 4 5
jj: 6 pp: 60 ii: 0 1 2 3 4 5
jj: 6 pp: 61 ii: 0 1 2 3 4 5
jj: 6 pp: 62 ii: 0 1 2 3 4 5
jj: 6 pp: 63 ii: 0 1 2 3 4 5

jj: 7 pp: 56 ii: 0 1 2 3 4 5 6
jj: 7 pp: 57 ii: 0 1 2 3 4 5 6
jj: 7 pp: 58 ii: 0 1 2 3 4 5 6
jj: 7 pp: 59 ii: 0 1 2 3 4 5 6
jj: 7 pp: 60 ii: 0 1 2 3 4 5 6
jj: 7 pp: 61 ii: 0 1 2 3 4 5 6
jj: 7 pp: 62 ii: 0 1 2 3 4 5 6
jj: 7 pp: 63 ii: 0 1 2 3 4 5 6
*/

// Create the buffers on each node
void DISTRIBUTED_ALLOCATE_NAME(int m0, int n0, float **A_distributed, float **B_distributed, float **C_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{

		*A_distributed = (float *)malloc(sizeof(float) * m0 * m0);
		*C_distributed = (float *)malloc(sizeof(float) * m0 * n0);
		*B_distributed = (float *)malloc(sizeof(float) * m0 * n0);
	}
	else
	{
		/*
	  STUDENT_TODO: Modify this is you plan to use more
	  than 1 rank to do work in distributed memory context.

	  Note: In the original configuration only rank with
	  rid == 0 has all of its buffers allocated.
		*/
	}
}

void DISTRIBUTE_DATA_NAME(int m0, int n0, float *A_sequential, float *B_sequential, float *A_distributed,
			  float *B_distributed)
{

	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	// Layout for sequential data
	// A is column major
	int rs_AS = m0;
	int cs_AS = 1;

	// B is column major
	int rs_BS = m0;
	int cs_BS = 1;

	// Note: Here is a perfect opportunity to change the layout
	//       of your data which has the potential to give you
	//       a sizeable performance gain.
	// Layout for distributed data
	// A is column major
	int rs_AD = m0;
	int cs_AD = 1;

	// B is column major
	int rs_BD = m0;
	int cs_BD = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{
		// Distribute the inputs
		for (int i0 = 0; i0 < m0; ++i0)
			for (int p0 = 0; p0 < m0; ++p0)
			{
				A_distributed[i0 * cs_AD + p0 * rs_AD] = A_sequential[i0 * cs_AS + p0 * rs_AS];
			}

		// Distribute the weights
		for (int p0 = 0; p0 < m0; ++p0)
			for (int j0 = 0; j0 < n0; ++j0)
			{
				B_distributed[p0 * cs_BD + j0 * rs_BD] = B_sequential[p0 * cs_BS + j0 * rs_BS];
			}
	}
	else
	{
		/*
	  STUDENT_TODO: Modify this is you plan to use more
	  than 1 rank to do work in distributed memory context.

	  Note: In the original configuration only rank with
	  rid == 0 has all of the necessary data for the computation.
	  All other ranks have garbage in their data. This is where
	  rank with rid == 0 needs to SEND data to the other nodes
	  to RECEIVE the data, or use COLLECTIVE COMMUNICATION to
	  distribute the data.
		*/
	}
}

void COLLECT_DATA_NAME(int m0, int n0, float *C_distributed, float *C_sequential)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	// Layout for sequential data
	// A is column major
	// C is column major
	int rs_CS = m0;
	int cs_CS = 1;

	// Note: Here is a perfect opportunity to change the layout
	//       of your data which has the potential to give you
	//       a sizeable performance gain.
	// Layout for distributed data
	// C is column major
	int rs_CD = m0;
	int cs_CD = 1;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{

		// Collect the output
		for (int i0 = 0; i0 < m0; ++i0)
			for (int j0 = 0; j0 < n0; ++j0)
				C_sequential[i0 * cs_CS + j0 * rs_CS] = C_distributed[i0 * cs_CD + j0 * rs_CD];
	}
	else
	{
		/*
	  STUDENT_TODO: Modify this is you plan to use more
	  than 1 rank to do work in distributed memory context.

	  Note: In the original configuration only rank with
	  rid == 0 performs the computation and copies the
	  "distributed" data to the "sequential" buffer that
	  is checked by the verifier on rank rid == 0. If the
	  other ranks contributed to the computation, then
	  rank rid == 0 needs to RECEIVE the contributions that
	  the other ranks SEND, or use COLLECTIVE COMMUNICATIONS
	  for the same result.
		*/
	}
}

void DISTRIBUTED_FREE_NAME(int m0, int n0, float *A_distributed, float *B_distributed, float *C_distributed)
{
	int rid;
	int num_ranks;
	int tag = 0;
	MPI_Status status;
	int root_rid = 0;

	MPI_Comm_rank(MPI_COMM_WORLD, &rid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

	if (rid == root_rid)
	{

		free(A_distributed);
		free(B_distributed);
		free(C_distributed);
	}
	else
	{
		/*
	  STUDENT_TODO: Modify this is you plan to use more
	  than 1 rank to do work in distributed memory context.

	  Note: In the original configuration only rank with
	  rid == 0 allocates the "distributed" buffers for itself.
	  If the other ranks were modified to allocate their own
	  buffers then they need to be freed at the end.
		*/
	}
}
