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

#include <math.h>
#include <string.h>
#include "timer.h"

int main(int argc, char** argv)
{
    int n = 4096;
    int m = 4096;
    int iter_max = 1000;
    
    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float error     = 1.0f;
    
    float A[n][m];
    float Anew[n][m];
    float y0[n];

    memset(A, 0, n * m * sizeof(float));
    
    // set boundary conditions
    for (int i = 0; i < m; i++)
    {
        A[0][i]   = 0.f;
        A[n-1][i] = 0.f;
    }
    
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
        A[j][0] = y0[j];
        A[j][m-1] = y0[j]*expf(-pi);
    }
    
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, m);
    
    StartTimer();
    int iter = 0;
    
#pragma omp parallel for shared(Anew)
    for (int i = 1; i < m; i++)
    {
       Anew[0][i]   = 0.f;
       Anew[n-1][i] = 0.f;
    }
#pragma omp parallel for shared(Anew)    
    for (int j = 1; j < n; j++)
    {
        Anew[j][0]   = y0[j];
        Anew[j][m-1] = y0[j]*expf(-pi);
    }
    
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }
        
#pragma omp parallel for shared(m, n, Anew, A)
#pragma acc kernels
        for( int j = 1; j < n-1; j++)
        {
            for( int i = 1; i < m-1; i++ )
            {
                A[j][i] = Anew[j][i];    
            }
        }

        if(iter % 100 == 0) printf("%5d, %0.6f\n", iter, error);
        
        iter++;
    }

    double runtime = GetTimer();
 
    printf(" total: %f s\n", runtime / 1000.f);
}
