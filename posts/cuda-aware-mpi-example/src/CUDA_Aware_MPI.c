// =======================================================================
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
// =======================================================================

#include "Jacobi.h"

/**
 * @file CUDA_Aware_MPI.c
 * @brief The implementation details for the CUDA-aware MPI version
 */

/**
 * @brief This allows the MPI process to set the CUDA device before the MPI environment is initialized
 * For the CUDA-aware MPI version, the is the only place where the device gets set. In order to
 * do this, we rely on the node's local rank, as the MPI environment has not been initialized yet.
 */
void SetDeviceBeforeInit()
{
	char * localRankStr = NULL;
	int rank = 0, devCount = 0;

	// We extract the local rank initialization using an environment variable
	if ((localRankStr = getenv(ENV_LOCAL_RANK)) != NULL)
	{
		rank = atoi(localRankStr);		
	}

	SafeCudaCall(cudaGetDeviceCount(&devCount));
	SafeCudaCall(cudaSetDevice(rank % devCount));
}

/**
 * @brief This allows the MPI process to set the CUDA device after the MPI environment is initialized
 * For the CUDA-aware MPI version, there is nothing to be done here.
 *
 * @param[in]	rank	The global rank of the calling MPI process
 */
void SetDeviceAfterInit(int rank)
{
}

/**
 * @brief Exchange halo values between 2 direct neighbors
 * This is the main difference between the normal CUDA & MPI version and the CUDA-aware MPI version. In the former, the exchange
 * first requires a copy from device to host memory, an MPI call using the host buffer and lastly, a copy of the received host buffer
 * back to the device memory. In the latter, the host buffers are completely skipped, as the MPI environment uses the device buffers
 * directly.
 *
 * @param[in]	cartComm		The carthesian MPI communicator
 * @param[in]	devSend			The device buffer that needs to be sent
 * @param[in]	hostSend		The host buffer where the device buffer is first copied to (not needed here)
 * @param[in]	hostRecv		The host buffer that receives the halo values (not needed here)
 * @param[in]	devRecv			The device buffer that receives the halo buffers directly
 * @param[in]	neighbor		The rank of the neighbor MPI process in the carthesian communicator
 * @param[in]	elemCount		The number of elements to transfer
 */
void ExchangeHalos(MPI_Comm cartComm, real * devSend, real * hostSend, real * hostRecv, real * devRecv, int neighbor, int elemCount)
{
	MPI_Status status;

	if (neighbor != MPI_PROC_NULL)
	{
		// Here it is not necessary to use host buffers and intermediate memory transfers
		MPI_Sendrecv(devSend, elemCount, MPI_CUSTOM_REAL, neighbor, 0, 
					 devRecv, elemCount, MPI_CUSTOM_REAL, neighbor, 0, cartComm, &status);
		SafeCheckMPIStatus(&status, elemCount);
	}
}
