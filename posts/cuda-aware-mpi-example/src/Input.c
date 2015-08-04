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
#include "Jacobi.h"

/**
 * @file Input.c
 * @brief This contains the command-line argument parser and support functions
 */

// ====================================
// Command-line arguments parsing block
// ====================================

// Print the usage information for this application
void PrintUsage(const char * appName)
{
	printf("Usage: %s -t Top.X [Top.Y] [-d Dom.X [Dom.Y]] [-fs] [-h | --help]\n", appName);
	printf(" -t Top.x [Top.y]: set the topology size (if \"Top.y\" is missing, the topology will default to (Top.x, 1); Top.x and Top.y must be positive integers)\n");
	printf(" -d Dom.x [Dom.y]: set the domain size per node (if \"Dom.y\" is missing, the domain size will default to (Dom.x, Dom.x); Dom.x and Dom.y must be positive integers)\n");
	printf(" -fs: use fast block swap after every iteration (this speeds up computation)\n");
	printf(" -h | --help: print help information\n");
}

// Find (and if found, erase) an argument in the command line
int FindAndClearArgument(const char * argName, int argc, char ** argv)
{
	for(int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], argName) == 0)
		{
			strcpy(argv[i], "");
			return i;
		}
	}

	return -1;
}

// Extract a number given as a command-line argument
int ExtractNumber(int argIdx, int argc, char ** argv)
{
	int result = 0;

	if (argIdx < argc)
	{
		result = atoi(argv[argIdx]);
		if (result > 0)
		{
			strcpy(argv[argIdx], "");
		}
	}

	return result;
}

/**
 * @brief Parses the application's command-line arguments
 *
 * @param[in] argc          The number of input arguments
 * @param[in] argv        	The input arguments
 * @param[in] rank          The MPI rank of the calling process
 * @param[in] size          The total number of MPI processes available
 * @param[out] domSize  	The parsed domain size (2D)
 * @param[out] topSize		The parsed topology size (2D)
 * @param[out] useFastSwap	The parsed flag for fast block swap
 * @return		     		The parsing status (STATUS_OK indicates a successful parse)
 */
int ParseCommandLineArguments(int argc, char ** argv, int rank, int size, int2 * domSize, int2 * topSize, int * useFastSwap)
{
	int canPrint = (rank == MPI_MASTER_RANK);
	int argIdx;

	// If help is requested, all other arguments will be ignored
	if ((FindAndClearArgument("-h", argc, argv) != -1) || (FindAndClearArgument("--help", argc, argv) != -1))
	{
		if (canPrint)
		{
			PrintUsage(argv[0]);
		}

		// This simply prevents the application from continuing 
		return STATUS_ERR;
	}

	// Check if fast swapping was requested
	* useFastSwap = (FindAndClearArgument("-fs", argc, argv) != -1);

	// Topology information must always be present
	argIdx = FindAndClearArgument("-t", argc, argv);
	if (argIdx == -1)
	{
		OneErrPrintf(canPrint, "Error: Could not find the topology information.\n");
		return STATUS_ERR;
	}
	else
	{
		topSize->x = ExtractNumber(argIdx + 1, argc, argv);
		topSize->y = ExtractNumber(argIdx + 2, argc, argv);

		// At least the first topology dimension must be specified
		if (topSize->x <= 0)
		{
			OneErrPrintf(canPrint, "Error: The topology size is invalid (first value: %d)\n", topSize->x);
			return STATUS_ERR;
		}

		// If the second topology dimension is missing, it will default to 1
		if (topSize->y <= 0)
		{
			topSize->y = 1;
		}
	}

	// The domain size information is optional
	argIdx = FindAndClearArgument("-d", argc, argv);
	if (argIdx == -1)
	{
		domSize->x = domSize->y = DEFAULT_DOMAIN_SIZE;
	}
	else
	{
		domSize->x = ExtractNumber(argIdx + 1, argc, argv);
		domSize->y = ExtractNumber(argIdx + 2, argc, argv);

		// At least the first domain dimension must be specified
		if (domSize->x < MIN_DOM_SIZE)
		{
			OneErrPrintf(canPrint, "Error: The local domain size must be at least %d (currently: %d)\n", MIN_DOM_SIZE, domSize->x);
			return STATUS_ERR;
		}
		
		// If the second domain dimension is missing, it will default to the first dimension's value
		if (domSize->y <= 0)
		{
			domSize->y = domSize->x;
		}
	}

	// At the end, there should be no other arguments that haven't been parsed
	for (int i = 1; i < argc; ++i)
	{
		if (strlen(argv[i]) > 0)
		{
			OneErrPrintf(canPrint, "Error: Unknown argument (\"%s\")\n", argv[i]);
			return STATUS_ERR;
		}
	}

	// If we reach this point, all arguments were parsed successfully
	if (canPrint)
	{
		printf("Topology size: %d x %d\n", topSize->x, topSize->y);
		printf("Local domain size (current node): %d x %d\n", domSize->x, domSize->y);
		printf("Global domain size (all nodes): %d x %d\n", topSize->x * domSize->x, topSize->y * domSize->y);
	}

	return STATUS_OK;
}

