/* Copyright (c) 1993-2018, NVIDIA CORPORATION. All rights reserved.
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
#ifndef __IO_HELPER_H__
#define __IO_HELPER_H__

#include <NvInfer.h>
#include <iostream>
#include <vector>

namespace nvinfer1
{

std::ostream& operator<<(std::ostream& o, const ILogger::Severity severity);

class Logger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, const char* msg) override
    {
        std::cerr << severity << ": " << msg << std::endl;
    }
};

template <typename T>
struct Destroy
{
    void operator()(T* t) const
    {
        t->destroy();
    }
};

std::string getBasename(std::string const& path);

size_t readTensor(std::vector<std::string> const& tensorProtoPaths, std::vector<float>& buffer);

void writeBuffer(void* buffer, size_t size, std::string const& path);

std::string readBuffer(std::string const& path);

} // namespace nvinfer1

#endif /*__IO_HELPER_H__*/
