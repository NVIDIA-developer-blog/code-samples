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

#include "Jacobi.h"

/**
 * @file Jacobi.c
 * @brief This contains the application entry point
 */

/**
 * @brief The application entry point
 *
 * @param[in] argc	The number of command-line arguments
 * @param[in] argv	The list of command-line arguments
 */
int main(int argc, char ** argv)
{
	MPI_Comm cartComm = MPI_COMM_WORLD;
	int rank, size;
	int neighbors[4];
	int2 domSize, topSize, topIndex;	
	int useFastSwap;
	
	double timerStart = 0.0;			
	double avgTransferTime = 0.0;
	int iterations = 0;
	
	real * devBlocks[2] 	= {NULL, NULL};		
	real * devSideEdges[2] 	= {NULL, NULL}; 	
	real * devHaloLines[2] 	= {NULL, NULL};
	real * hostSendLines[2] = {NULL, NULL};
	real * hostRecvLines[2] = {NULL, NULL};
	real * devResidue 		= NULL;

	cudaStream_t copyStream = NULL;
	
	// Initialize the MPI process
	Initialize(&argc, &argv, &rank, &size);

	// Extract topology and domain dimensions from the command-line arguments
	if (ParseCommandLineArguments(argc, argv, rank, size, &domSize, &topSize, &useFastSwap) == STATUS_OK)
	{	
		// Map the MPI ranks to corresponding GPUs and the desired topology
		if (ApplyTopology(&rank, size, &topSize, neighbors, &topIndex, &cartComm) == STATUS_OK)
		{
			// Initialize the data for the current MPI process
			InitializeDataChunk(topSize.y, topIndex.y, &domSize, neighbors, &copyStream, devBlocks, devSideEdges, 
				devHaloLines, hostSendLines, hostRecvLines, &devResidue);
			
			// Run the Jacobi computation
			PreRunJacobi(cartComm, rank, size, &timerStart);
			RunJacobi(cartComm, rank, size, &domSize, &topIndex, neighbors, useFastSwap, devBlocks, devSideEdges, 
				devHaloLines, hostSendLines, hostRecvLines, devResidue, copyStream, &iterations, &avgTransferTime);
			PostRunJacobi(cartComm, rank, size, &topSize, &domSize, iterations, useFastSwap, timerStart, avgTransferTime);
		}
	}

	// Finalize the MPI process
	Finalize(devBlocks, devSideEdges, devHaloLines, hostSendLines, hostRecvLines, devResidue, copyStream);

	return STATUS_OK;
}
