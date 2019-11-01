/* Copyright (c) 1993-2019, NVIDIA CORPORATION. All rights reserved.
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
const express = require('express')
const app = express()

const kernelSrc = `
__global__ void mandelbrot(int *img, int width_pixel, int height_pixel,
                           float w, float h, float x0, float y0, int max_iter) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  float c_re = (w / width_pixel) * (x - width_pixel / 2) + x0;
  float c_im = (h / height_pixel) * (height_pixel / 2 - y) + y0;
  float z_re = 0, z_im = 0;
  int iter = 0;
  while ((z_re * z_re + z_im * z_im <= 4) && (iter < max_iter)) {
    float z_re_new = z_re * z_re - z_im * z_im + c_re;
    z_im = 2 * z_re * z_im + c_im;
    z_re = z_re_new;
    iter += 1;
  }
  img[y * width_pixel + x] = (iter == max_iter);
}`
const port = 3000
const widthPixels = 128
const heightPixels = 64
const buildkernel = Polyglot.eval('grcuda', 'buildkernel')
const DeviceArray = Polyglot.eval('grcuda', 'DeviceArray')
const kernel = buildkernel(kernelSrc, 'mandelbrot',
  'pointer, sint32, sint32, float, float, float, float, sint32')
const blockSize = [32, 8]  // thread block with 32x8 threads
const grid = [widthPixels / blockSize[0], heightPixels / blockSize[1]]
const kernelWithGrid = kernel(grid, blockSize)

app.get('/', (req, res) => {
  const img = DeviceArray('int', heightPixels, widthPixels)
  kernelWithGrid(img, widthPixels, heightPixels, 3.0, 2.0, -0.5, 0.0, 255)
  var textImg = ''
  for (var y = 0; y < heightPixels; y++) {
    for (var x = 0; x < widthPixels; x++) {
      textImg += (img[y][x] === 1) ? '*' : ' '
    }
    textImg += '\n'
  }
  res.setHeader('Content-Type', 'text/plain')
  res.send(textImg)
})

app.listen(port, () => console.log(`Mandelbrot app listening on port ${port}!`))
