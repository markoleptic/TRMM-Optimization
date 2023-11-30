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

	const int block_size = 8;

	if (rid == root_rid)
	{
		for (int j0 = 0; j0 < n0; j0 += block_size)
		{
			for (int i0 = 0; i0 <= j0; i0 += block_size)
			{
				for (int jj = j0; jj < MIN(j0 + block_size, n0); ++jj)
				{
					for (int ii = i0; ii < MIN(i0 + block_size, jj); ++ii)
					{
						float res = 0.0f;
						for (int p0 = 0; p0 < m0; ++p0)
						{
							float A_ip = A_distributed[ii * cs_A + p0 * rs_A];
							float B_pj = B_distributed[p0 * cs_B + jj * rs_B];
							res += A_ip * B_pj;
						}
						C_distributed[ii * cs_C + jj * rs_C] = res;
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

// Access Pattern when block_size = 8:
/*
jj: 0 ii:
jj: 1 ii: 0
jj: 2 ii: 0 1
jj: 3 ii: 0 1 2
jj: 4 ii: 0 1 2 3
jj: 5 ii: 0 1 2 3 4
jj: 6 ii: 0 1 2 3 4 5
jj: 7 ii: 0 1 2 3 4 5 6
jj: 8 ii: 0 1 2 3 4 5 6 7
jj: 9 ii: 0 1 2 3 4 5 6 7
jj: 10 ii: 0 1 2 3 4 5 6 7
jj: 11 ii: 0 1 2 3 4 5 6 7
jj: 12 ii: 0 1 2 3 4 5 6 7
jj: 13 ii: 0 1 2 3 4 5 6 7
jj: 14 ii: 0 1 2 3 4 5 6 7
jj: 15 ii: 0 1 2 3 4 5 6 7
jj: 8 ii:
jj: 9 ii: 8
jj: 10 ii: 8 9
jj: 11 ii: 8 9 10
jj: 12 ii: 8 9 10 11
jj: 13 ii: 8 9 10 11 12
jj: 14 ii: 8 9 10 11 12 13
jj: 15 ii: 8 9 10 11 12 13 14
jj: 16 ii: 0 1 2 3 4 5 6 7
jj: 17 ii: 0 1 2 3 4 5 6 7
jj: 18 ii: 0 1 2 3 4 5 6 7
jj: 19 ii: 0 1 2 3 4 5 6 7
jj: 20 ii: 0 1 2 3 4 5 6 7
jj: 21 ii: 0 1 2 3 4 5 6 7
jj: 22 ii: 0 1 2 3 4 5 6 7
jj: 23 ii: 0 1 2 3 4 5 6 7
jj: 16 ii: 8 9 10 11 12 13 14 15
jj: 17 ii: 8 9 10 11 12 13 14 15
jj: 18 ii: 8 9 10 11 12 13 14 15
jj: 19 ii: 8 9 10 11 12 13 14 15
jj: 20 ii: 8 9 10 11 12 13 14 15
jj: 21 ii: 8 9 10 11 12 13 14 15
jj: 22 ii: 8 9 10 11 12 13 14 15
jj: 23 ii: 8 9 10 11 12 13 14 15
jj: 16 ii:
jj: 17 ii: 16
jj: 18 ii: 16 17
jj: 19 ii: 16 17 18
jj: 20 ii: 16 17 18 19
jj: 21 ii: 16 17 18 19 20
jj: 22 ii: 16 17 18 19 20 21
jj: 23 ii: 16 17 18 19 20 21 22
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
