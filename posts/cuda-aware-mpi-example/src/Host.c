/* Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string.h>
#include <math.h>
#include "Jacobi.h"

/**
 * @file Host.c
 * @brief This contains the host functions for data allocations, message passing and host-side computations
 */

// =================
// Support functions
// =================

// Allocate a block of host memory
real * SafeHostAlloc(size_t byteCount)
{
	real * newBuffer = (real *)malloc(byteCount);

	if (newBuffer == NULL)
	{
		fprintf(stderr, "Error: Failed to allocate host memory (%lu bytes). Terminating...\n", byteCount);
		exit(STATUS_ERR);
	}

	return newBuffer;
}

// Display a number in a pretty format
char * FormatNumber(double value, const char * suffix, char * printBuf)
{
	char * magnitude = " kMGT";
	int orderIdx = 0;
	
	value = fabs(value);
	while ((value > 1000.0) && (orderIdx < strlen(magnitude) - 1))
	{
		++orderIdx;
		value /= 1000.0;
	}
	
	sprintf(printBuf, "%.2lf %c%s", value, magnitude[orderIdx], suffix);

	return printBuf;
}

// Swap 2 device buffers
void SwapDeviceBlocks(real * devBlocks[2])
{
	real * tempBlock = devBlocks[0];

	devBlocks[0] = devBlocks[1];
	devBlocks[1] = tempBlock;
}

// Check the status of an MPI transfer
void SafeCheckMPIStatus(MPI_Status * status, int expectedElems)
{
	int recvElems;

	MPI_Get_count(status, MPI_CUSTOM_REAL, &recvElems);

	if (recvElems != expectedElems)
	{
		fprintf(stderr, "Error: MPI transfer returned %d elements, but %d were expected. "
			"Terminating...\n", recvElems, expectedElems);
		exit(STATUS_ERR);
	}
}

// ===================================
// MPI initialization and finalization
// ===================================

/**
 * @brief Initialize the MPI environment, allowing the CUDA device to be selected before (if necessary)
 *
 * @param[in, out]	argc	The number of command-line arguments
 * @param[in, out]	argv	The list of command-line arguments
 * @param[out]		rank	The global rank of the current MPI process
 * @param[out]		size	The total number of MPI processes available
 */
void Initialize(int * argc, char *** argv, int * rank, int * size)
{
	// Setting the device here will have an effect only for the CUDA-aware MPI version
	SetDeviceBeforeInit();

	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, rank);
	MPI_Comm_size(MPI_COMM_WORLD, size);
}

/**
 * @brief Close (finalize) the MPI environment and deallocate buffers
 *
 * @param[in]	devBlocks		The 2 device blocks that were used during the Jacobi iterations
 * @param[in]	devSideEdges	The 2 device side edges that were used to hold updated halos before sending
 * @param[in]	devHaloLines	The 2 device lines that were used to hold received halos
 * @param[in]	hostSendLines	The 2 host send buffers that were used at halo exchange in the normal CUDA & MPI version
 * @param[in]	hostRecvLines	The 2 host receive buffers that were used at halo exchange in the normal CUDA & MPI version
 * @param[in]	devResidue		The global residue, kept in device memory	
 * @param[in]	copyStream		The stream used to overlap top & bottom halo exchange with side halo copy to host memory
 */
void Finalize(real * devBlocks[2], real * devSideEdges[2], real * devHaloLines[2], real * hostSendLines[2], 
		real * hostRecvLines[2], real * devResidue, cudaStream_t copyStream)
{
	MPI_Finalize();

	for(int i = 0; i < 2; ++i)
	{
		SafeHostFree(hostSendLines[i]);
		SafeHostFree(hostRecvLines[i]);
		SafeDevFree(devHaloLines[i]);
		SafeDevFree(devBlocks[i]);
		SafeDevFree(devSideEdges[i]);
	}

	SafeDevFree(devResidue);	
	if (copyStream != NULL)
	{
		SafeCudaCall(cudaStreamDestroy(copyStream));
	}	
}

// ====================
// Topology application
// ====================

/**
 * @brief Generates the 2D topology and establishes the neighbor relationships between MPI processes
 *
 * @param[in, out]  rank		The rank of the calling MPI process
 * @param[in]  		size		The total number of MPI processes available
 * @param[in]  		topSize		The desired topology size (this must match the number of available MPI processes)
 * @param[out] 		neighbors	The list that will be populated with the direct neighbors of the calling MPI process
 * @param[out] 		topIndex	The 2D index that the calling MPI process will have in the topology
 * @param[out]		cartComm	The carthesian MPI communicator
 */
int ApplyTopology(int * rank, int size, const int2 * topSize, int * neighbors, int2 * topIndex, MPI_Comm * cartComm)
{
	int topologySize = topSize->x * topSize->y;
	int dimSize[2] = {topSize->x, topSize->y};
	int usePeriods[2] = {0, 0}, newCoords[2];
	int oldRank = * rank;
	
	// The number of MPI processes must fill the topology
	if (size != topologySize)
	{
		OneErrPrintf(* rank == MPI_MASTER_RANK, "Error: The number of MPI processes (%d) doesn't match "
				"the topology size (%d).\n", size, topologySize);
		
		return STATUS_ERR;
	}

	// Create a carthesian communicator
	MPI_Cart_create(MPI_COMM_WORLD, 2, dimSize, usePeriods, 1, cartComm);

	// Update the rank to be relevant to the new communicator
	MPI_Comm_rank(* cartComm, rank);

	if ((* rank) != oldRank)
	{
		printf("Rank change: from %d to %d\n", oldRank, * rank);
	}

	// Obtain the 2D coordinates in the new communicator
	MPI_Cart_coords(* cartComm, * rank, 2, newCoords);
	* topIndex = make_int2(newCoords[0], newCoords[1]);

	// Obtain the direct neighbor ranks
	MPI_Cart_shift(* cartComm, 0, 1, neighbors + DIR_LEFT, neighbors + DIR_RIGHT);
	MPI_Cart_shift(* cartComm, 1, 1, neighbors + DIR_TOP, neighbors + DIR_BOTTOM);

	// Setting the device here will have effect only for the normal CUDA & MPI version
	SetDeviceAfterInit(* rank);

	return STATUS_OK;
}

// ===================
// Data initialization
// ===================

// Initialize the send and receive buffers for a given direction
void InitExchangeBuffers(real * hostSendLines[2], real * hostRecvLines[2], int bufIndex, size_t byteCount)
{
	hostSendLines[bufIndex] = SafeHostAlloc(byteCount);
	hostRecvLines[bufIndex] = SafeHostAlloc(byteCount);
}

/**
 * @brief This allocates and initializes all the relevant data buffers before the Jacobi run
 *
 * @param[in]	topSizeY		The size of the topology in the Y direction
 * @param[in]	topIdxY			The Y index of the calling MPI process in the topology
 * @param[in]	domSize			The size of the local domain (for which only the current MPI process is responsible)
 * @param[in]	neighbors		The neighbor ranks, according to the topology
 * @param[in]	copyStream		The stream used to overlap top & bottom halo exchange with side halo copy to host memory
 * @param[out]	devBlocks		The 2 device blocks that will be updated during the Jacobi run
 * @param[out]	devSideEdges	The 2 side edges (parallel to the Y direction) that will hold the packed halo values before sending them
 * @param[out]	devHaloLines	The 2 halo lines (parallel to the Y direction) that will hold the packed halo values after receiving them
 * @param[out] 	hostSendLines	The 2 host send buffers that will be used during the halo exchange by the normal CUDA & MPI version
 * @param[out]	hostRecvLines	The 2 host receive buffers that will be used during the halo exchange by the normal CUDA & MPI version
 * @param[out]	devResidue		The global device residue, which will be updated after every Jacobi iteration
 */
void InitializeDataChunk(int topSizeY, int topIdxY, const int2 * domSize, const int * neighbors, cudaStream_t * copyStream, 
		real * devBlocks[2], real * devSideEdges[2], real * devHaloLines[2], real * hostSendLines[2], real * hostRecvLines[2], real ** devResidue)
{
	const real PI = (real)3.1415926535897932384626;
	const real E_M_PI = (real)exp(-PI);
	
	size_t blockBytes = (domSize->x + 2) * (domSize->y + 2) * sizeof(real);
	size_t sideLineBytes = domSize->y * sizeof(real);
	int2 borderBounds = make_int2(topIdxY * domSize->y, (topIdxY + 1) * domSize->y);
	int borderSpan = domSize->y * topSizeY - 1;
	real * hostBlock = SafeHostAlloc(blockBytes);

	// Clearing the block also sets the boundary conditions for top and bottom edges to 0
	memset(hostBlock, 0, blockBytes);

	InitExchangeBuffers(hostSendLines, hostRecvLines, 0, domSize->x * sizeof(real));
	InitExchangeBuffers(hostSendLines, hostRecvLines, 1, sideLineBytes);

	// Set the boundary conditions for the left edge
	if (!HasNeighbor(neighbors, DIR_LEFT))
	{
		for (int j = borderBounds.x, idx = domSize->x + 3; j < borderBounds.y; ++j, idx += domSize->x + 2)
		{
			hostBlock[idx] = (real)sin(PI * j / borderSpan);
		}
	}

	// Set the boundary conditions for the right edge
	if (!HasNeighbor(neighbors, DIR_RIGHT))
	{
		for (int j = borderBounds.x, idx = ((domSize->x + 2) << 1) - 2; j < borderBounds.y; ++j, idx += domSize->x + 2)
		{
			hostBlock[idx] = (real)sin(PI * j / borderSpan) * E_M_PI;
		}
	}

	// Perform device memory allocation and initialization
	for (int i = 0; i < 2; ++i)
	{
		SafeCudaCall(cudaMalloc((void **)&devBlocks[i], blockBytes));
		SafeCudaCall(cudaMalloc((void **)&devSideEdges[i], sideLineBytes));	
		SafeCudaCall(cudaMalloc((void **)&devHaloLines[i], sideLineBytes));

		SafeCudaCall(cudaMemset(devSideEdges[i], 0, sideLineBytes));
	}

	SafeCudaCall(cudaMalloc((void **)devResidue, sizeof(real)));
	SafeCudaCall(cudaMemcpy(devBlocks[0], hostBlock, blockBytes, cudaMemcpyHostToDevice));
	SafeCudaCall(cudaMemcpy(devBlocks[1], devBlocks[0], blockBytes, cudaMemcpyDeviceToDevice));
	SafeCudaCall(cudaStreamCreate(copyStream));			

	SafeHostFree(hostBlock);
}

// =====================
// Jacobi implementation
// =====================

// Update the performance counters for the current MPI process
void UpdatePerfCounters(const int2 * topSize, const int2 * domSize, int iterations, int useFastSwap, 
		double * lattUpdates, double * flops, double * bandWidth)
{
	* lattUpdates = 1.0 * (topSize->x * domSize->x - 2) * (topSize->y * domSize->y - 2) * iterations;
	* flops = 5.0 * (* lattUpdates);							// Operations per Jacobi kernel run
	* bandWidth = 6.0 * (* lattUpdates) * sizeof(real);			// Transfers per Jacobi kernel run

	if (!useFastSwap)
	{
		* bandWidth += 2.0 * (* lattUpdates) * sizeof(real);	// Transfers from block copying must be included
	}
}

// Print a performance counter in a specific format
void PrintPerfCounter(const char * counterDesc, const char * counterUnit, double counter, double elapsedTime, int size)
{
	char printBuf[256];
	double avgCounter = counter / elapsedTime;
	double rankAvgCounter = avgCounter / size;

	printf("%s: %s (total), ", counterDesc, FormatNumber(avgCounter, counterUnit, printBuf));
	printf("%s (per process)\n", FormatNumber(rankAvgCounter, counterUnit, printBuf));
}

/**
 * @brief This function is called immediately before the main Jacobi loop
 *
 * @param[in]	cartComm	The carthesian communicator
 * @param[in]	rank		The rank of the calling MPI process
 * @param[in]	size		The total number of MPI processes available
 * @param[out]	timerStart	The Jacobi loop starting moment (measured as wall-time)
 */
void PreRunJacobi(MPI_Comm cartComm, int rank, int size, double * timerStart)
{
	struct cudaDeviceProp devProps;
	int crtDevice = 0, enabledECC = 0;

	// We get the properties of the current device, assuming all other devices are the same
	SafeCudaCall(cudaGetDevice(&crtDevice));
	SafeCudaCall(cudaGetDeviceProperties(&devProps, crtDevice));

	// Determine how many devices have ECC enabled (assuming exactly one process per device)
	MPI_Reduce(&devProps.ECCEnabled, &enabledECC, 1, MPI_INT, MPI_SUM, MPI_MASTER_RANK, cartComm);

	MPI_Barrier(cartComm);
	OnePrintf(rank == MPI_MASTER_RANK, "Starting Jacobi run with %d processes using \"%s\" GPUs (ECC enabled: %d / %d):\n", 
				size, devProps.name, enabledECC, size);
	* timerStart = MPI_Wtime();
}

/**
 * @brief This function is called immediately after the main Jacobi loop
 *
 * @param[in]	cartComm		The carthesian communicator
 * @param[in]	rank			The rank of the calling MPI process
 * @param[in]	topSize			The size of the topology
 * @param[in]	domSize			The size of the local domain
 * @param[in]	iterations		The number of successfully completed Jacobi iterations
 * @param[in]	useFastSwap		The flag indicating if fast pointer swapping was used to exchange blocks
 * @param[in]	timerStart		The Jacobi loop starting moment (measured as wall-time)
 * @param[in]	avgTransferTime	The average time spent performing MPI transfers (per process)
 */
void PostRunJacobi(MPI_Comm cartComm, int rank, int size, const int2 * topSize, const int2 * domSize, int iterations, int useFastSwap, 
			double timerStart, double avgTransferTime)
{
	double elapsedTime;
	double lattUpdates = 0.0, flops = 0.0, bandWidth = 0.0;

	MPI_Barrier(cartComm);
	elapsedTime = MPI_Wtime() - timerStart;
		
	// Show the performance counters
	if (rank == MPI_MASTER_RANK)
	{
		printf("Total Jacobi run time: %.4lf sec.\n", elapsedTime);
		printf("Average per-process communication time: %.4lf sec.\n", avgTransferTime);

		// Compute the performance counters over all MPI processes
		UpdatePerfCounters(topSize, domSize, iterations, useFastSwap, &lattUpdates, &flops, &bandWidth);
	
		PrintPerfCounter("Measured lattice updates", "LU/s", lattUpdates, elapsedTime, size);
		PrintPerfCounter("Measured FLOPS", "FLOPS", flops, elapsedTime, size);
		PrintPerfCounter("Measured device bandwidth", "B/s", bandWidth, elapsedTime, size);
	}
}

/**
 * @brief This performs the exchanging of all necessary halos between 2 neighboring MPI processes
 *
 * @param[in]		cartComm		The carthesian MPI communicator
 * @param[in]		domSize			The 2D size of the local domain
 * @param[in]		topIndex		The 2D index of the calling MPI process in the topology
 * @param[in]		neighbors		The list of ranks which are direct neighbors to the caller
 * @param[in]		copyStream		The stream used to overlap top & bottom halo exchange with side halo copy to host memory
 * @param[in, out]	devBlocks		The 2 device blocks that are updated during the Jacobi run
 * @param[in, out]	devSideEdges	The 2 side edges (parallel to the Y direction) that hold the packed halo values before sending them
 * @param[in, out]	devHaloLines	The 2 halo lines (parallel to the Y direction) that hold the packed halo values after receiving them
 * @param[in, out] 	hostSendLines	The 2 host send buffers that are used during the halo exchange by the normal CUDA & MPI version
 * @param[in, out]	hostRecvLines	The 2 host receive buffers that are used during the halo exchange by the normal CUDA & MPI version
 * @return							The time spent during the MPI transfers
 */
double TransferAllHalos(MPI_Comm cartComm, const int2 * domSize, const int2 * topIndex, const int * neighbors, cudaStream_t copyStream,
	real * devBlocks[2], real * devSideEdges[2], real * devHaloLines[2], real * hostSendLines[2], real * hostRecvLines[2])
{
	real * devSendLines[2] = {devBlocks[0] + domSize->x + 3, devBlocks[0] + domSize->y * (domSize->x + 2) + 1};
	real * devRecvLines[2] = {devBlocks[0] + 1, devBlocks[0] + (domSize->y + 1) * (domSize->x + 2) + 1};
	int yNeighbors[2] = {neighbors[DIR_TOP], neighbors[DIR_BOTTOM]};
	int xNeighbors[2] = {neighbors[DIR_LEFT], neighbors[DIR_RIGHT]};
	int2 order = make_int2(topIndex->x % 2, topIndex->y % 2);
	double transferTime;

	// Populate the block's side edges
	CopyDevSideEdgesFromBlock(devBlocks[0], devSideEdges, domSize, neighbors, copyStream);

	// Exchange data with the top and bottom neighbors
	transferTime = MPI_Wtime();
	ExchangeHalos(cartComm, devSendLines[order.y], hostSendLines[0], 
		hostRecvLines[0], devRecvLines[order.y], yNeighbors[order.y], domSize->x);
	ExchangeHalos(cartComm, devSendLines[1 - order.y], hostSendLines[0], 
		hostRecvLines[0], devRecvLines[1 - order.y], yNeighbors[1 - order.y], domSize->x);

	SafeCudaCall(cudaStreamSynchronize(copyStream));
	
	// Exchange data with the left and right neighbors
	ExchangeHalos(cartComm, devSideEdges[order.x], hostSendLines[1], 
		hostRecvLines[1], devHaloLines[order.x], xNeighbors[order.x], domSize->y);
	ExchangeHalos(cartComm, devSideEdges[1 - order.x], hostSendLines[1], 
		hostRecvLines[1], devHaloLines[1 - order.x], xNeighbors[1 - order.x], domSize->y); 
	transferTime = MPI_Wtime() - transferTime;

	// Copy the received halos to the device block
	CopyDevHalosToBlock(devBlocks[0], devHaloLines[0], devHaloLines[1], domSize, neighbors);

	return transferTime;
}

// Get the bounds of the chunk area that n
int4 GetComputeBounds(const int2 * size, const int * neighbors)
{
	return make_int4(neighbors[DIR_LEFT] 	!= MPI_PROC_NULL? 0 : 1,
					 neighbors[DIR_TOP] 	!= MPI_PROC_NULL? 0 : 1,
					 neighbors[DIR_RIGHT]   != MPI_PROC_NULL? size->x - 1 : size->x - 2,
					 neighbors[DIR_BOTTOM]  != MPI_PROC_NULL? size->y - 1 : size->y - 2);
}

/**
 * @brief This is the main Jacobi loop, which handles device computation and data exchange between MPI processes
 *
 * @param[in]		cartComm		The carthesian MPI communicator
 * @param[in]		rank			The rank of the calling MPI process
 * @param[in]		size			The number of available MPI processes
 * @param[in]		domSize			The 2D size of the local domain
 * @param[in]		topIndex		The 2D index of the calling MPI process in the topology
 * @param[in]		neighbors		The list of ranks which are direct neighbors to the caller
 * @param[in]		useFastSwap		This flag indicates if blocks should be swapped through pointer copy (faster) or through element-by-element copy (slower)
 * @param[in, out]	devBlocks		The 2 device blocks that are updated during the Jacobi run
 * @param[in, out]	devSideEdges	The 2 side edges (parallel to the Y direction) that hold the packed halo values before sending them
 * @param[in, out]	devHaloLines	The 2 halo lines (parallel to the Y direction) that hold the packed halo values after receiving them
 * @param[in, out] 	hostSendLines	The 2 host send buffers that are used during the halo exchange by the normal CUDA & MPI version
 * @param[in, out]	hostRecvLines	The 2 host receive buffers that are used during the halo exchange by the normal CUDA & MPI version
 * @param[in, out]	devResidue		The global device residue, which gets updated after every Jacobi iteration
 * @param[in]		copyStream		The stream used to overlap top & bottom halo exchange with side halo copy to host memory
 * @param[out]		iterations		The number of successfully completed iterations
 * @param[out]		avgTransferTime The average time spent performing MPI transfers (per process)
 */
void RunJacobi(MPI_Comm cartComm, int rank, int size, const int2 * domSize, const int2 * topIndex, const int * neighbors, int useFastSwap,
	real * devBlocks[2], real * devSideEdges[2], real * devHaloLines[2], real * hostSendLines[2], real * hostRecvLines[2], real * devResidue,
	cudaStream_t copyStream, int * iterations, double * avgTransferTime)
{
	real residue, globalResidue = (real)1.0;
	int4 bounds = GetComputeBounds(domSize, neighbors);
	double localTime = 0.0;
	
	* iterations = 0;
	* avgTransferTime = 0.0;

	while ((* iterations < JACOBI_MAX_LOOPS) && (globalResidue > JACOBI_TOLERANCE))
	{
		// Compute the residue for the current iteration
		residue = CallJacobiKernel(devBlocks, devResidue, &bounds, domSize);
		
		// Exchange the old block with the new (updated) one
		if (useFastSwap)
		{
			SwapDeviceBlocks(devBlocks);
		}
		else
		{
			CopyDeviceBlock(devBlocks, &bounds, domSize);
		}

		// Send and receive halo (exchange) elements
		localTime += TransferAllHalos(cartComm, domSize, topIndex, neighbors, copyStream, devBlocks, devSideEdges, devHaloLines, hostSendLines, hostRecvLines);

		// Obtain and distribute the global maximum residue
		globalResidue = (real)0.0;
		MPI_Allreduce(&residue, &globalResidue, 1, MPI_CUSTOM_REAL, MPI_MAX, cartComm);

		OnePrintf((rank == MPI_MASTER_RANK) && ((* iterations) % 100 == 0),
			"Iteration: %d - Residue: %.6f\n", * iterations, globalResidue);

		++(* iterations);
	}

	// Calculate the total time spent on transfers
	MPI_Reduce(&localTime, avgTransferTime, 1, MPI_DOUBLE, MPI_SUM, MPI_MASTER_RANK, cartComm);
	* avgTransferTime /= size;

	OnePrintf(rank == MPI_MASTER_RANK, "Stopped after %d iterations with residue %.6f\n", * iterations, globalResidue);
}

