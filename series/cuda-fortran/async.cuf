! Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
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

module kernels_m
contains
  attributes(global) subroutine kernel(a, offset)
    implicit none
    real :: a(*)
    integer, value :: offset
    integer :: i
    real :: c, s, x

    i = offset + threadIdx%x + (blockIdx%x-1)*blockDim%x
    x = i; s = sin(x); c = cos(x)
    a(i) = a(i) + sqrt(s**2+c**2)
  end subroutine kernel
end module kernels_m


program testAsync
  use cudafor
  use kernels_m
  implicit none
  integer, parameter :: blockSize = 256, nStreams = 4
  integer, parameter :: n = 4*1024*blockSize*nStreams
  real, pinned, allocatable :: a(:)
  real, device :: a_d(n)
  integer(kind=cuda_stream_kind) :: stream(nStreams)
  type (cudaEvent) :: startEvent, stopEvent, dummyEvent
  real :: time
  integer :: i, istat, offset, streamSize = n/nStreams
  logical :: pinnedFlag
  type (cudaDeviceProp) :: prop
  
  istat = cudaGetDeviceProperties(prop, 0)
  write(*,"(' Device: ', a,/)") trim(prop%name)

  ! allocate pinned  host memory
  allocate(a(n), STAT=istat, PINNED=pinnedFlag)
  if (istat /= 0) then
     write(*,*) 'Allocation of a failed'
     stop
  else
     if (.not. pinnedFlag) write(*,*) 'Pinned allocation failed'
  end if

  ! create events and streams
  istat = cudaEventCreate(startEvent)
  istat = cudaEventCreate(stopEvent)  
  istat = cudaEventCreate(dummyEvent)  
  do i = 1, nStreams
     istat = cudaStreamCreate(stream(i))
  enddo
  
  ! baseline case - sequential transfer and execute
  a = 0
  istat = cudaEventRecord(startEvent,0)
  a_d = a
  call kernel<<<n/blockSize, blockSize>>>(a_d, 0)
  a = a_d
  istat = cudaEventRecord(stopEvent, 0)
  istat = cudaEventSynchronize(stopEvent)
  istat = cudaEventElapsedTime(time, startEvent, stopEvent)
  write(*,*) 'Time for sequential transfer and execute (ms): ', time
  write(*,*) '  max error: ', maxval(abs(a-1.0))

  ! asynchronous version 1: loop over {copy, kernel, copy}
  a = 0
  istat = cudaEventRecord(startEvent,0)
  do i = 1, nStreams
     offset = (i-1)*streamSize
     istat = cudaMemcpyAsync(a_d(offset+1),a(offset+1),streamSize,stream(i))
     call kernel<<<streamSize/blockSize, blockSize, &
                   0, stream(i)>>>(a_d,offset)
     istat = cudaMemcpyAsync(a(offset+1),a_d(offset+1),streamSize,stream(i))
  enddo
  istat = cudaEventRecord(stopEvent, 0)
  istat = cudaEventSynchronize(stopEvent)
  istat = cudaEventElapsedTime(time, startEvent, stopEvent)
  write(*,*) 'Time for asynchronous V1 transfer and execute (ms): ', time
  write(*,*) '  max error: ', maxval(abs(a-1.0))

  ! asynchronous version 2: 
  ! loop over copy, loop over kernel, loop over copy
  a = 0
  istat = cudaEventRecord(startEvent,0)
  do i = 1, nStreams
     offset = (i-1)*streamSize
     istat = cudaMemcpyAsync(a_d(offset+1),a(offset+1),streamSize,stream(i))
  enddo
  do i = 1, nStreams
     offset = (i-1)*streamSize
     call kernel<<<streamSize/blockSize, blockSize, &
                   0, stream(i)>>>(a_d,offset)
  enddo
  do i = 1, nStreams
     offset = (i-1)*streamSize
     istat = cudaMemcpyAsync(a(offset+1),a_d(offset+1),streamSize,stream(i))
  enddo
  istat = cudaEventRecord(stopEvent, 0)
  istat = cudaEventSynchronize(stopEvent)
  istat = cudaEventElapsedTime(time, startEvent, stopEvent)
  write(*,*) 'Time for asynchronous V2 transfer and execute (ms): ', time
  write(*,*) '  max error: ', maxval(abs(a-1.0))

  ! cleanup
  istat = cudaEventDestroy(startEvent)
  istat = cudaEventDestroy(stopEvent)
  istat = cudaEventDestroy(dummyEvent)
  do i = 1, nStreams
     istat = cudaStreamDestroy(stream(i))
  enddo
  deallocate(a)

end program testAsync

