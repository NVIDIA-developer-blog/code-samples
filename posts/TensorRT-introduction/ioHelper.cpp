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
#include <algorithm>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iterator>
#include <onnx/onnx_pb.h>
#include "ioHelper.h"
using namespace std;

namespace nvinfer1
{

string getBasename(string const& path)
{
#ifdef _WIN32
    constexpr char SEPARATOR = '\\';
#else
    constexpr char SEPARATOR = '/';
#endif
    int baseId = path.rfind(SEPARATOR) + 1;
    return path.substr(baseId, path.rfind('.') - baseId);
}

ostream& operator<<(ostream& o, const nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
    case ILogger::Severity::kINTERNAL_ERROR: o << "INTERNAL_ERROR"; break;
    case ILogger::Severity::kERROR: o << "ERROR"; break;
    case ILogger::Severity::kWARNING: o << "WARNING"; break;
    case ILogger::Severity::kINFO: o << "INFO"; break;
    }
    return o;
}

// returns number of floats successfully read from tensor protobuf
size_t readTensorProto(string const& path, float* buffer)
{
    string const data{readBuffer(path)};
    onnx::TensorProto tensorProto;
    if (!tensorProto.ParseFromString(data))
        return 0;

    assert(tensorProto.has_raw_data());
    assert(tensorProto.raw_data().size() % sizeof(float) == 0);

    memcpy(buffer, tensorProto.raw_data().data(), tensorProto.raw_data().size());
    return tensorProto.raw_data().size() / sizeof(float);
}

// returns number of floats successfully read from tensorProtoPaths
size_t readTensor(vector<string> const& tensorProtoPaths, vector<float>& buffer)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    size_t totalElements = 0;

    for (size_t i = 0; i < tensorProtoPaths.size(); ++i)
    {
        size_t elements = readTensorProto(tensorProtoPaths[i], &buffer[totalElements]);        if (!elements)
        {
            cout << "ERROR: could not read tensor from file " << tensorProtoPaths[i] << endl;
            break;
        }
        totalElements += elements;
    }

    return totalElements;
}

void writeBuffer(void* buffer, size_t size, string const& path)
{
    ofstream stream(path.c_str(), ios::binary);

    if (stream)
        stream.write(static_cast<char*>(buffer), size);
}

// Returns empty string iff can't read the file
string readBuffer(string const& path)
{
    string buffer;
    ifstream stream(path.c_str(), ios::binary);

    if (stream)
    {
        stream >> noskipws;
        copy(istream_iterator<char>(stream), istream_iterator<char>(), back_inserter(buffer));
    }

    return buffer;
}

} // namespace nvinfer1
