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
 * @file CUDA_Normal_MPI.c
 * @brief The implementation details for the normal CUDA & MPI version
 */

/**
 * @brief This allows the MPI process to set the CUDA device before the MPI environment is initialized
 * For the normal CUDA & MPI version, the device will be set later on, so this implementation does nothing.
 */
void SetDeviceBeforeInit()
{
}

/**
 * @brief This allows the MPI process to set the CUDA device after the MPI environment is initialized
 * For the normal CUDA & MPI version, this is the only place where the MPI process actually sets the CUDA device. Since there can
 * be more than one MPI process working with a given CUDA device, actually selecting the device is done using the global rank. This
 * will yield best results (the least GPU contention) if the ranks are consecutive.
 *
 * @param[in]	rank	The global rank of the calling MPI process
 */
void SetDeviceAfterInit(int rank)
{
	int devCount = 0;	

	SafeCudaCall(cudaGetDeviceCount(&devCount));
	SafeCudaCall(cudaSetDevice(rank % devCount));
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
 * @param[in]	hostSend		The host buffer where the device buffer is first copied to
 * @param[in]	hostRecv		The host buffer that receives the halo values directly
 * @param[in]	devRecv			The device buffer where the receiving host buffer is copied to
 * @param[in]	neighbor		The rank of the neighbor MPI process in the carthesian communicator
 * @param[in]	elemCount		The number of elements to transfer
 */
void ExchangeHalos(MPI_Comm cartComm, real * devSend, real * hostSend, real * hostRecv, real * devRecv, int neighbor, int elemCount)
{
	size_t byteCount = elemCount * sizeof(real);
	MPI_Status status;

	if (neighbor != MPI_PROC_NULL)
	{
		SafeCudaCall(cudaMemcpy(hostSend, devSend, byteCount, cudaMemcpyDeviceToHost));
		MPI_Sendrecv(hostSend, elemCount, MPI_CUSTOM_REAL, neighbor, 0, 
					 hostRecv, elemCount, MPI_CUSTOM_REAL, neighbor, 0, cartComm, &status);
		SafeCheckMPIStatus(&status, elemCount);
		SafeCudaCall(cudaMemcpy(devRecv, hostRecv, byteCount, cudaMemcpyHostToDevice));
	}
}
