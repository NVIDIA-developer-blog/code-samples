/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <cstdio>

#include <nvml.h>
#define NVML_CALL( call )				\
{										\
	nvmlReturn_t nvmlError = call;		\
	if (NVML_SUCCESS != nvmlError )	{	\
		fprintf (stderr, "NVML_ERROR: %s (%d) in %d line of %s\n", nvmlErrorString( nvmlError ), nvmlError , __LINE__, __FILE__ ); \
	}									\
}

/**
 * getNvmlDevice determines the NVML Device Id of the currently active CUDA device
 *
 * @param[out]  nvmlDeviceId    the NVML Device Id of the currently active CUDA device
 * @return                      NVML_SUCCESS in case of success. Error code of NVML API
 *                              or NVML_ERROR_UNKNOWN if CUDA Runtime API failed otherwise
 */
inline nvmlReturn_t getNvmlDevice( nvmlDevice_t* nvmlDeviceId )
{
	int activeCUDAdevice = 0;
	cudaError_t cudaError = cudaGetDevice ( &activeCUDAdevice );
	if ( cudaSuccess  != cudaError )
		return NVML_ERROR_UNKNOWN;
	
	cudaDeviceProp activeCUDAdeviceProp;
	cudaError = cudaGetDeviceProperties ( &activeCUDAdeviceProp, activeCUDAdevice );
	if ( cudaSuccess  != cudaError )
		return NVML_ERROR_UNKNOWN;
	
	unsigned int nvmlDeviceCount = 0;
	nvmlReturn_t nvmlError = nvmlDeviceGetCount ( &nvmlDeviceCount );
	if ( NVML_SUCCESS != nvmlError )
		return nvmlError;
	
	for ( unsigned int nvmlDeviceIdx = 0; nvmlDeviceIdx < nvmlDeviceCount; ++nvmlDeviceIdx )
	{
		nvmlError = nvmlDeviceGetHandleByIndex ( nvmlDeviceIdx, nvmlDeviceId );
		if ( NVML_SUCCESS != nvmlError )
			return nvmlError; 
		nvmlPciInfo_t nvmPCIInfo;
		nvmlError = nvmlDeviceGetPciInfo ( *nvmlDeviceId, &nvmPCIInfo );
		if ( NVML_SUCCESS != nvmlError )
			return nvmlError;
		if ( static_cast<unsigned int>(activeCUDAdeviceProp.pciBusID) == nvmPCIInfo.bus &&
		     static_cast<unsigned int>(activeCUDAdeviceProp.pciDeviceID) == nvmPCIInfo.device &&
			 static_cast<unsigned int>(activeCUDAdeviceProp.pciDomainID) == nvmPCIInfo.domain )
			break;
	}
	return NVML_SUCCESS;
}

inline nvmlReturn_t reportApplicationClocks( nvmlDevice_t nvmlDeviceId )
{
	unsigned int appSMclock = 0;
	unsigned int appMemclock = 0;
	nvmlReturn_t nvmlError = nvmlDeviceGetApplicationsClock ( nvmlDeviceId, NVML_CLOCK_SM, &appSMclock );
	if ( NVML_SUCCESS != nvmlError )
			return nvmlError;
	nvmlError = nvmlDeviceGetApplicationsClock ( nvmlDeviceId, NVML_CLOCK_MEM, &appMemclock );
	if ( NVML_SUCCESS != nvmlError )
			return nvmlError;
	
	std::cout<<"Application Clocks = ("<<appMemclock<<","<<appSMclock<<")"<<std::endl;
	return NVML_SUCCESS;
}

int matrixMultiply(dim3 &dimsA, dim3 &dimsB);

int main()
{
	cudaSetDevice(0);
	cudaFree(0);
	
	NVML_CALL( nvmlInit() );
	
	nvmlDevice_t nvmlDeviceId;
	NVML_CALL( getNvmlDevice( &nvmlDeviceId ) );
	
	unsigned int memClock = 0;
	NVML_CALL( nvmlDeviceGetClockInfo( nvmlDeviceId, NVML_CLOCK_MEM, &memClock ) );
	
	unsigned int numSupportedSMClocks = 32;
	unsigned int smClocksMHz[32];
	NVML_CALL( nvmlDeviceGetSupportedGraphicsClocks ( nvmlDeviceId, memClock, &numSupportedSMClocks, smClocksMHz ) );
	
	unsigned int numSupportedMemClocks = 32;
	unsigned int memClocksMHz[32];
	NVML_CALL( nvmlDeviceGetSupportedMemoryClocks ( nvmlDeviceId, &numSupportedMemClocks, memClocksMHz ) ); 

	unsigned int maxSMclock = 0;
	unsigned int maxMemclock = 0;
	NVML_CALL( nvmlDeviceGetMaxClockInfo ( nvmlDeviceId, NVML_CLOCK_SM, &maxSMclock ) );
	NVML_CALL( nvmlDeviceGetMaxClockInfo ( nvmlDeviceId, NVML_CLOCK_MEM, &maxMemclock ) );

	//Check permissions to modify application clocks
	nvmlEnableState_t isRestricted;
	NVML_CALL( nvmlDeviceGetAPIRestriction ( nvmlDeviceId, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, &isRestricted ) );
	
	if ( NVML_FEATURE_DISABLED == isRestricted )
	{
		dim3 dimsA(1024,1024);
		dim3 dimsB(1024,1024);
		
		for ( int i=numSupportedSMClocks-1; i >= 0; --i )
		{
			std::cout<<"Setting ";
			NVML_CALL( nvmlDeviceSetApplicationsClocks ( nvmlDeviceId, memClocksMHz[0], smClocksMHz[i] ) );
			
			NVML_CALL( reportApplicationClocks( nvmlDeviceId ) );
			
			matrixMultiply(dimsA, dimsB);
		}
	}
	else
	{
		std::cerr<<"ERROR: Application clock permissions are set to RESTRICTED. Please change with sudo nvidia-smi -acp UNRESTRICTED"<<std::endl;
	}
	
	//Reset Application Clocks and Shutdown NVML
	if ( NVML_FEATURE_DISABLED == isRestricted )
	{
		NVML_CALL( nvmlDeviceResetApplicationsClocks ( nvmlDeviceId ) ); 
	}
	NVML_CALL( nvmlShutdown() );
	
	cudaDeviceReset();
	return 0;
}
