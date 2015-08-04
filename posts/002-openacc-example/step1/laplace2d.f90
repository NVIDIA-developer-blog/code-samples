! Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!  * Redistributions of source code must retain the above copyright
!    notice, this list of conditions and the following disclaimer.
!  * Redistributions in binary form must reproduce the above copyright
!    notice, this list of conditions and the following disclaimer in the
!    documentation and/or other materials provided with the distribution.
!  * Neither the name of NVIDIA CORPORATION nor the names of its
!    contributors may be used to endorse or promote products derived
!    from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
! EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
! PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
! CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
! EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
! PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
! PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
! OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
! (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

program laplace
  implicit none
  integer, parameter :: fp_kind=kind(1.0)
  integer, parameter :: n=4096, m=4096, iter_max=1000
  integer :: i, j, iter
  real(fp_kind), dimension (:,:), allocatable :: A, Anew
  real(fp_kind), dimension (:),   allocatable :: y0
  real(fp_kind) :: pi=2.0_fp_kind*asin(1.0_fp_kind), tol=1.0e-5_fp_kind, error=1.0_fp_kind
  real(fp_kind) :: start_time, stop_time

  allocate ( A(0:n-1,0:m-1), Anew(0:n-1,0:m-1) )
  allocate ( y0(0:m-1) )

  A = 0.0_fp_kind

  ! Set B.C.
  y0 = sin(pi* (/ (j,j=0,m-1) /) /(m-1))

  A(0,:)   = 0.0_fp_kind
  A(n-1,:) = 0.0_fp_kind
  A(:,0)   = y0
  A(:,m-1) = y0*exp(-pi)
   
  write(*,'(a,i5,a,i5,a)') 'Jacobi relaxation Calculation:', n, ' x', m, ' mesh'
 
  call cpu_time(start_time) 

  iter=0

!$omp parallel do shared(Anew)
  do i=1,m-1
    Anew(0,i)   = 0.0_fp_kind
    Anew(n-1,i) = 0.0_fp_kind
  end do
!$end omp parallel do
!$omp parallel do shared(Anew)
  do i=1,n-1
    Anew(i,0)   = y0(i)
    Anew(i,m-1) = y0(i)*exp(-pi)
  end do
!$end omp parallel do

  do while ( error .gt. tol .and. iter .lt. iter_max )
    error=0.0_fp_kind

!$omp parallel shared(m, n, Anew, A)

!$omp do reduction( max:error )
!$acc kernels
    do j=1,m-2
      do i=1,n-2
        Anew(i,j) = 0.25_fp_kind * ( A(i+1,j  ) + A(i-1,j  ) + &
                                     A(i  ,j-1) + A(i  ,j+1) )
        error = max( error, abs(Anew(i,j)-A(i,j)) )
      end do
    end do
!$acc end kernels
!$omp end do

    if(mod(iter,100).eq.0 ) write(*,'(i5,f10.6)'), iter, error
    iter = iter +1

!$omp do
!$acc kernels
    do j=1,m-2
      do i=1,n-2
        A(i,j) = Anew(i,j)
      end do
    end do
!$acc end kernels
!$omp end do

!$omp end parallel

  end do

  call cpu_time(stop_time) 
  write(*,'(a,f10.3,a)')  ' completed in ', stop_time-start_time, ' seconds'

  deallocate (A,Anew,y0)
end program laplace
