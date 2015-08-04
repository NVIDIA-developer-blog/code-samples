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
 * @file Device.cu
 * @brief This contains the device kernels as well as the host wrappers for these kernels
 */

// ================================================
// Templates for handling "float" and "double" data
// ================================================

// The default implementation for atomic maximum
template <typename T>
__device__ void AtomicMax(T * const address, const T value)
{
	atomicMax(address, value);
}

/**
 * @brief Compute the maximum of 2 single-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
template <>
__device__ void AtomicMax(float * const address, const float value)
{
	if (* address >= value)
	{
		return;
	}

	int * const address_as_i = (int *)address;
	int old = * address_as_i, assumed;

	do 
	{
		assumed = old;
		if (__int_as_float(assumed) >= value)
		{
			break;
		}

		old = atomicCAS(address_as_i, assumed, __float_as_int(value));
	} while (assumed != old);
}

/**
 * @brief Compute the maximum of 2 double-precision floating point values using an atomic operation
 *
 * @param[in]	address	The address of the reference value which might get updated with the maximum
 * @param[in]	value	The value that is compared to the reference in order to determine the maximum
 */
template <>
__device__ void AtomicMax(double * const address, const double value)
{
	if (* address >= value)
	{
		return;
	}

	uint64 * const address_as_i = (uint64 *)address;
    uint64 old = * address_as_i, assumed;

	do 
	{
        assumed = old;
		if (__longlong_as_double(assumed) >= value)
		{
			break;
		}
		
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

// Templates for computing the absolute value of a real number
template<typename T> 	__device__ T 		rabs(T val) 		{ return abs(val);   }
template<> 				__device__ float 	rabs(float val) 	{ return fabsf(val); }
template<> 				__device__ double 	rabs(double val) 	{ return fabs(val);  }

// ==============
// Device kernels
// ==============

/**
 * @brief The device kernel for copying (unpacking) the values from the halo buffers to the left and right side of the data block
 *
 * @param[out] 	block				The 2D device block that will contain the halo values after unpacking
 * @param[in] 	haloLeft			The halo buffer for the left side of the data block
 * @param[in]	haloRight			The halo buffer for the right side of the data block
 * @param[in] 	size				The 2D size of data block, excluding the edges which hold the halo values
 * @param[in]	hasLeftNeighbor		Marks if the calling MPI process has a left neighbor
 * @param[in]	hasRightNeighbor	Marks if the calling MPI process has a right neighbor
 */
__global__ void HaloToBufferKernel(real * __restrict__ const block, const real * __restrict__ const haloLeft,
	const real * __restrict__ const haloRight, int2 size, int hasLeftNeighbor, int hasRightNeighbor)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size.y)
	{
		int start = (idx + 1) * (size.x + 2);

		if (hasLeftNeighbor)
		{
			block[start] = haloLeft[idx];
		}
		
		if (hasRightNeighbor)
		{
			block[start + size.x + 1] = haloRight[idx];
		}
	}
}

/**
 * @brief The device kernel for copying (packing) the values on the left and right side of the data block to separate, contiguous buffers.
 * 
 * @param[in] 	block				The 2D device block containing the updated values after a Jacobi run
 * @param[out] 	leftSideEdge		The left edge, holding the left-most updated halos
 * @param[out]	rightSideEdge		The right edge, holding the right-most updated halos
 * @param[in] 	size				The 2D size of data block, excluding the edges which hold the halo values for the next iteration
 * @param[in]	hasLeftNeighbor		Marks if the calling MPI process has a left neighbor
 * @param[in]	hasRightNeighbor	Marks if the calling MPI process has a right neighbor
 */
__global__ void BufferToHaloKernel(const real * __restrict__ const block, real * __restrict__ const leftSideEdge,
								   real * __restrict__ const rightSideEdge, int2 size, int hasLeftNeighbor, int hasRightNeighbor)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size.y)
	{
		int start = (idx + 1) * (size.x + 2);

		if (hasLeftNeighbor)
		{
			leftSideEdge[idx] = block[start];
		}

		if (hasRightNeighbor)
		{
			rightSideEdge[idx] = block[start + size.x + 1];
		}
	}
}

/**
 * @brief The device kernel for one Jacobi iteration
 * 
 * @param[in]	oldBlock	The current (old) block providing the values at the start of the iteration
 * @param[out]	newBlock	The new block that will hold the updated values after the iteration finishes
 * @param[in] 	bounds		The bounds of the rectangular block region that holds only computable values
 * @param[in] 	stride		The stride of the data blocks, excluding the edges holding the halo values
 * @param[out]	devResidue	The global residue that is to be updated through the iteration
 */
__global__ void JacobiComputeKernel(const real * __restrict__ const oldBlock, real * __restrict__ const newBlock,
					int4 bounds, int stride, real * devResidue)
{
	int2 idx = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int memIdx = (idx.y + 1) * (stride + 2) + idx.x + 1;

	if ((idx.x < bounds.x) || (idx.x > bounds.z) || (idx.y < bounds.y) || (idx.y > bounds.w))
	{
		return;
	}
	
	// Each thread computes its new value based on the neighbors' old values
	real newVal = ((real)0.25) * (oldBlock[memIdx - 1] + oldBlock[memIdx + 1] + 
						  oldBlock[memIdx - stride - 2] + oldBlock[memIdx + stride + 2]);

	// The computed elements are written to the new buffer
	newBlock[memIdx] = newVal;

	// The global maximum residue must be updated
	AtomicMax<real>(devResidue, rabs(newVal - oldBlock[memIdx]));
}

/**
 * @brief The host wrapper for copying the updated block over the old one, after a Jacobi iteration finishes
 * 
 * @param[in] 	srcBlock	The source block, from which data will be read
 * @param[out] 	dstBlock	The destination block, where data will be written
 * @param[in] 	bounds		The bounds of the rectangular updated region (holding only computable values)
 * @param[in] 	size		The stride of the data blocks, excluding the halo values
 */
__global__ void CopyBlockKernel(const real * __restrict__ const srcBlock, real * __restrict__ const dstBlock, int4 bounds, int stride)
{
	int2 idx = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int memIdx = (idx.y + 1) * (stride + 2) + idx.x + 1;

	if ((idx.x < bounds.x) || (idx.x > bounds.z) || (idx.y < bounds.y) || (idx.y > bounds.w))
	{
		return;
	}

	dstBlock[memIdx] = srcBlock[memIdx];
}

// =============
// Host wrappers
// =============

/**
 * @brief The host function for checking the result of a CUDA API call
 * 
 * @param[in]	command			The result of the previously-issued CUDA API call 
 * @param[in]	commandName		The name of the issued API call 
 * @param[in]	fileName		The name of the file where the API call occurred
 * @param[in]	line			The line in the file where the API call occurred
 */
extern "C" void CheckCudaCall(cudaError_t command, const char * commandName, const char * fileName, int line)
{
	if (command != cudaSuccess)
	{
		fprintf(stderr, "Error: CUDA result \"%s\" for call \"%s\" in file \"%s\" at line %d. Terminating...\n", 
			cudaGetErrorString(command), commandName, fileName, line);
		exit(STATUS_ERR);
	}
}

/**
 * @brief The host wrapper for one Jacobi iteration
 * 
 * @param[in, out]	devBlocks	The 2 blocks involved: the first is the current one, the second is the one to be updated
 * @param[out]		devResidue	The global residue that is to be updated through the Jacobi iteration
 * @param[in] 		bounds		The bounds of the rectangular block region that holds only computable values
 * @param[in] 		size		The 2D size of data blocks, excluding the edges which hold the halo values
 */
extern "C" real CallJacobiKernel(real * devBlocks[2], real * devResidue, const int4 * bounds, const int2 * size)
{
	const dim3 blockSize(16, 16, 1);
	const dim3 gridSize(1 + (size->x - 1) / 16, 1 + (size->y - 1) / 16, 1);
	real residue = (real)0.0;
	
	// Launch the kernel for the Jacobi iteration on the default stream
	SafeCudaCall(cudaMemcpy(devResidue, &residue, sizeof(real), cudaMemcpyHostToDevice));
	JacobiComputeKernel<<<gridSize, blockSize>>>(devBlocks[0], devBlocks[1], * bounds, size->x, devResidue);
	CheckCudaCall(cudaGetLastError(), "JacobiComputeKernel<<<,>>>(...)", __FILE__, __LINE__);

	// Finally, we obtain the global maximum residue (this is also a device synchronization point)
	SafeCudaCall(cudaMemcpy(&residue, devResidue, sizeof(real), cudaMemcpyDeviceToHost));

	return residue;
}

/**
 * @brief The host wrapper for copying the updated block over the old one, after a Jacobi iteration finishes
 * 
 * @param[in, out]	devBlocks	The 2 blocks involved: the first is the old one, the second is the updated one
 * @param[in] 		bounds		The bounds of the rectangular updated region (holding only computable values)
 * @param[in] 		size		The 2D size of data blocks, excluding the edges which hold the halo values
 */
extern "C" void CopyDeviceBlock(real * devBlocks[2], const int4 * bounds, const int2 * size)
{
	const dim3 copyBlockSize(16, 16, 1);
	const dim3 copyGridSize(1 + (size->x - 1) / 16, 1 + (size->y - 1) / 16, 1);

	CopyBlockKernel<<<copyGridSize, copyBlockSize>>>(devBlocks[1], devBlocks[0], * bounds, size->x);
	CheckCudaCall(cudaGetLastError(), "CopyBlockKernel<<<,>>>(...)", __FILE__, __LINE__);
	SafeCudaCall(cudaStreamSynchronize(NULL));
}

/**
 * @brief The host wrapper for copying (unpacking) the values from the halo buffers to the left and right side of the data block
 *
 * @param[out] 	devBlock			The 2D device block that will contain the halo values after unpacking
 * @param[in] 	devHaloLineLeft		The halo buffer for the left side of the data block
 * @param[in]	devHaloLineRight	The halo buffer for the right side of the data block
 * @param[in]	size				The 2D size of data block, excluding the edges which hold the halo values
 * @param[in]	neighbors			The ranks of the neighboring MPI processes
 */
extern "C" void CopyDevHalosToBlock(real * devBlock, const real * devHaloLineLeft, const real * devHaloLineRight, 
					const int2 * size, const int * neighbors)
{
	// Don't perform unpacking if there are no neighbors on the sides
	if (!HasNeighbor(neighbors, DIR_LEFT) && !HasNeighbor(neighbors, DIR_RIGHT))
	{
		return;
	}

	const int blockSize = 256;
	const int gridSize = 1 + (size->y - 1) / blockSize;	

	HaloToBufferKernel<<<gridSize, blockSize>>>(devBlock, devHaloLineLeft, devHaloLineRight, * size,
		HasNeighbor(neighbors, DIR_LEFT), HasNeighbor(neighbors, DIR_RIGHT));
	CheckCudaCall(cudaGetLastError(), "HaloToBufferKernel<<<,>>>(...)", __FILE__, __LINE__);
	SafeCudaCall(cudaStreamSynchronize(NULL));
}

/**
 * @brief The host wrapper for copying (packing) the values on the left and right side of the data block to separate, contiguous buffers
 * 
 * @param[in] 	devBlock		The 2D device block containing the updated values after a Jacobi run
 * @param[out] 	devSideEdges	The buffers where the edge values will be packed in
 * @param[in] 	size			The 2D size of data block, excluding the edges which hold the halo values for the next iteration
 * @param[in]	neighbors		The ranks of the neighboring MPI processes
 * @param[in]	copyStream		The stream on which this kernel will be executed
 */
extern "C" void CopyDevSideEdgesFromBlock(const real * devBlock, real * devSideEdges[2], const int2 * size, const int * neighbors, cudaStream_t copyStream)
{
	// Don't perform packing if there are no neighbors on the sides
	if (!HasNeighbor(neighbors, DIR_LEFT) && !HasNeighbor(neighbors, DIR_RIGHT))
	{
		return;
	}

	const int blockSize = 256;
	const int gridSize = 1 + (size->y - 1) / blockSize;	

	BufferToHaloKernel<<<gridSize, blockSize, 0, copyStream>>>(devBlock, devSideEdges[0], devSideEdges[1], * size,
		HasNeighbor(neighbors, DIR_LEFT), HasNeighbor(neighbors, DIR_RIGHT));
	CheckCudaCall(cudaGetLastError(), "BufferToHaloKernel<<<,>>>(...)", __FILE__, __LINE__);
}
