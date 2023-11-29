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

#include "utils.c"
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

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

	const int block_size = 128;

	if (rid == root_rid)
	{
		/* Initialize with 0 because the initial value will be random garbage,
		   necessary because using += operator in the i0 loop */
		for (int i0 = 0; i0 < n0; ++i0)
		{
			for (int p0 = 0; p0 < m0; ++p0)
			{
				C_distributed[i0 * rs_C + p0] = 0.0f;
			}
		}
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
						// This checks if along the diagonal or not. If the block (ii_max - i)
						// is large enough, it isn't on the diagonal, and we can proceed with
						// simd. Otherwise use default
						if (ii_max - i0 >= block_size)
						{
							for (int pp = p0; pp < pp_max; ++pp)
							{
								// "Broadcast" B values
								__m256 B_pj = _mm256_set1_ps(
								    B_distributed[pp * cs_B + jj * rs_B]);
								for (int ii = i0; ii < ii_max; ii += 8)
								{
									__m256 A_ip = _mm256_loadu_ps(
									    &A_distributed[ii * cs_A + pp * rs_A]);
									__m256 C = _mm256_loadu_ps(
									    &C_distributed[ii * cs_C + jj * rs_C]);
									C = _mm256_fmadd_ps(A_ip, B_pj, C);
									_mm256_storeu_ps(
									    &C_distributed[ii * cs_C + jj * rs_C], C);
								}
							}
						}
						else
						{
							for (int pp = p0; pp < pp_max; ++pp)
							{
								float B_pj = B_distributed[pp * cs_B + jj * rs_B];
								for (int ii = i0; ii < ii_max; ++ii)
								{
									float A_ip =
									    A_distributed[ii * cs_A + pp * rs_A];
									C_distributed[ii * cs_C + jj * rs_C] +=
									    A_ip * B_pj;
								}
							}
						}
					}
				}
			}
		}
	}
	else
	{
		/* STUDENT_TODO: Modify this is you plan to use more
		 than 1 rank to do work in distributed memory context. */
	}
}

// old code
// for (int j0 = 0; j0 < n0; j0 += block_size) {
//     int jj_max = MIN(j0 + block_size, n0);
//     for (int p0 = 0; p0 < m0; p0 += block_size) {
//         int pp_max = MIN(p0 + block_size, m0);
//         for (int i0 = 0; i0 <= j0; i0 += block_size) {
//             for (int jj = j0; jj < jj_max; ++jj) {
//                 int ii_max = MIN(i0 + block_size, jj);
//                 for (int pp = p0; pp < pp_max; ++pp) {
//                     __m256 B_pj = _mm256_set1_ps(B_distributed[pp * cs_B + jj * rs_B]);
//                     for (int ii = i0; ii < ii_max; ii += 8) {
//                         __m256 A_ip = _mm256_loadu_ps(&A_distributed[ii * cs_A + pp * rs_A]);
//                         __m256 C = _mm256_loadu_ps(&C_distributed[ii * cs_C + jj * rs_C]);
//                         C = _mm256_fmadd_ps(A_ip, B_pj, C);
//                         _mm256_storeu_ps(&C_distributed[ii * cs_C + jj * rs_C], C);
//                     }
//                 }
//             }
//         }
//     }
// }
// for (int j0 = 0; j0 < n0; j0 += block_size)
// {
//     int jj_max = MIN(j0 + block_size, n0);
// 	for (int p0 = 0; p0 < m0; p0 += block_size)
// 	{
//         int pp_max = MIN(p0 + block_size, m0);
// 		for (int i0 = 0; i0 <= j0; i0 += block_size)
// 		{
// 			for (int jj = j0; jj < jj_max; ++jj)
// 			{
//                 int ii_max = MIN(i0 + block_size, jj);
// 				if (MIN(i0 + 8, jj) - i0 > 8)
// 				{
// 					for (int pp = p0; pp < pp_max; ++pp)
// 					{
//                         __m256 B_pj = _mm256_set1_ps(B_distributed[pp * cs_B + jj * rs_B]);
// 						__m256 C = _mm256_loadu_ps(&C_distributed[i0 * cs_C + jj
// * rs_C]); 						for (int ii = i0; ii < ii_max; ii += 8)
// 						{
// 							__m256 A_ip = _mm256_loadu_ps(&A_distributed[i0 *
// cs_A
// + pp
// * rs_A]); 							C = _mm256_fmadd_ps(A_ip, B_pj, C);
// 						}
// 						_mm256_storeu_ps(&C_distributed[i0 * cs_C + jj * rs_C],
// C);
// 					}
// 				}
// 				else
// 				{
// 					for (int pp = p0; pp < pp_max; ++pp)
// 					{
// 						float B_pj = B_distributed[pp * cs_B + jj * rs_B];
// 						for (int ii = i0; ii < ii_max; ++ii)
// 						{
// 							float A_ip = A_distributed[ii * cs_A + pp *
// rs_A]; 							C_distributed[ii * cs_C + jj * rs_C] +=
// A_ip
// * B_pj;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }
// for (int j0 = 0; j0 < n0; j0 += block_size)
// {
// 	for (int p0 = 0; p0 < m0; p0 += block_size)
// 	{
// 		for (int i0 = 0; i0 <= j0; i0 += block_size)
// 		{
// 			for (int jj = j0; jj < MIN(j0 + block_size, n0); ++jj)
// 			{
//                 int ii_max = MIN(i0 + block_size, jj);
// 				if (MIN(i0 + 8, jj) - i0 > 8)
// 				{
//                     greaterThanEight++;
// 					for (int pp = p0; pp < MIN(p0 + block_size, m0); ++pp)
// 					{
// 						__m256 c0 = _mm256_loadu_ps(
// 						    &C_distributed[i0 * cs_C + jj * rs_C]);
// 						for (int ii = i0; ii < ii_max; ii += 8)
// 						{
// 							float B = B_distributed[pp * cs_B + jj * rs_B];
// 							__m256 a0 = _mm256_loadu_ps(
// 							    &A_distributed[i0 * cs_A + pp * rs_A]);
// 							c0 = _mm256_fmadd_ps(a0, _mm256_set1_ps(B), c0);
// 						}
// 						_mm256_storeu_ps(&C_distributed[i0 * cs_C + jj * rs_C],
// 								 c0);
// 					}
// 				}
// 				else
// 				{
//                     lessThanEight++;
// 					for (int pp = p0; pp < MIN(p0 + block_size, m0); ++pp)
// 					{
// 						float B_pj = B_distributed[pp * cs_B + jj * rs_B];
// 						for (int ii = i0; ii < ii_max; ++ii)
// 						{
// 							float A_ip =
// 							    A_distributed[ii * cs_A + pp * rs_A];
// 							C_distributed[ii * cs_C + jj * rs_C] +=
// 							    A_ip * B_pj;
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }
// if (m0 == 64)
// {
// 	float *array = malloc(sizeof(float) * m0 * n0);
// 	for (int i0 = 0; i0 < n0; ++i0)
// 	{
// 		for (int p0 = 0; p0 < m0; ++p0)
// 		{
// 			array[i0 * rs_C + p0] = 0.0f;
// 		}
// 	}
// 	for (int j0 = 0; j0 < n0; ++j0)
// 	{
// 		for (int p0 = 0; p0 < m0; ++p0)
// 		{
// 			float B = B_distributed[p0 * cs_B + j0 * rs_B];
// 			for (int i0 = 0; i0 < j0; ++i0)
// 			{
// 				float A = A_distributed[i0 * cs_A + p0 * rs_A];
// 				array[i0 * cs_C + j0 * rs_C] += A * B;
// 			}
// 		}
// 	}
// 	printDistributedDiff(array, C_distributed, m0 * n0, "test.txt");
// 	free(array);
// }

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