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
#include <NvInfer.h>
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <numeric>
#include <math.h>

using namespace nvinfer1;
using namespace std;
using namespace cudawrapper;

static Logger gLogger;

// Number of times we run inference to calculate average time.
constexpr int ITERATIONS = 10;
// Maxmimum absolute tolerance for output tensor comparison against reference.
constexpr double ABS_EPSILON = 0.005;
// Maxmimum relative tolerance for output tensor comparison against reference.
constexpr double REL_EPSILON = 0.05;
// Allow TensorRT to use up to 1GB of GPU memory for tactic selection.
constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30; // 1 GB

ICudaEngine* createCudaEngine(string const& onnxModelPath, int batchSize)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
    unique_ptr<nvinfer1::IBuilder, Destroy<nvinfer1::IBuilder>> builder{nvinfer1::createInferBuilder(gLogger)};
    unique_ptr<nvinfer1::INetworkDefinition, Destroy<nvinfer1::INetworkDefinition>> network{builder->createNetworkV2(explicitBatch)};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};
    unique_ptr<nvinfer1::IBuilderConfig,Destroy<nvinfer1::IBuilderConfig>> config{builder->createBuilderConfig()};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setMaxBatchSize(batchSize);
    
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT, Dims4{1, 3, 256 , 256});
    profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX, Dims4{32, 3, 256 , 256});    
    config->addOptimizationProfile(profile);

    return builder->buildEngineWithConfig(*network, *config);
}

static int getBindingInputIndex(IExecutionContext* context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
    int inputId = getBindingInputIndex(context);

    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueueV2(bindings, stream, nullptr);
    cudaMemcpyAsync(outputTensor.data(), bindings[1 - inputId], outputTensor.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

void doInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
    CudaEvent start;
    CudaEvent end;
    double totalTime = 0.0;

    for (int i = 0; i < ITERATIONS; ++i)
    {
        float elapsedTime;

        // Measure time it takes to copy input to GPU, run inference and move output back to CPU.
        cudaEventRecord(start, stream);
        launchInference(context, stream, inputTensor, outputTensor, bindings, batchSize);
        cudaEventRecord(end, stream);

        // Wait until the work is finished.
        cudaStreamSynchronize(stream);
        cudaEventElapsedTime(&elapsedTime, start, end);

        totalTime += elapsedTime;
    }

    cout << "Inference batch size " << batchSize << " average over " << ITERATIONS << " runs is " << totalTime / ITERATIONS << "ms" << endl;
}

void verifyOutput(vector<float> const& outputTensor, vector<float> const& referenceTensor, int size)
{
    for (size_t i = 0; i < size; ++i)
    {
        double reference = static_cast<double>(referenceTensor[i]);
        // Check absolute and relative tolerance.
        if (abs(outputTensor[i] - reference) > max(abs(reference) * REL_EPSILON, ABS_EPSILON))
        {
            cout << "ERROR: mismatch at position " << i;
            cout << " expected " << reference << ", but was " << outputTensor[i] << endl;
            return;
        }
    }

    cout << "OK" << endl;
}

int main(int argc, char* argv[])
{
    // Declaring cuda engine.
    unique_ptr<ICudaEngine, Destroy<ICudaEngine>> engine{nullptr};
    // Declaring execution context.
    unique_ptr<IExecutionContext, Destroy<IExecutionContext>> context{nullptr};
    vector<float> inputTensor;
    vector<float> outputTensor;
    vector<float> referenceTensor;
    void* bindings[2]{0};
    vector<string> inputFiles;
    CudaStream stream;

    if (argc < 3)
    {
        cout << "usage: " << argv[0] << " <path_to_model.onnx> (1.. <path_to_input.pb>)" << endl;
        return 1;
    }

    string onnxModelPath(argv[1]);
    for (int i = 2; i < argc; ++i)
        inputFiles.push_back(string{argv[i]});

    int batchSize = inputFiles.size();


    // Create Cuda Engine.
    engine.reset(createCudaEngine(onnxModelPath, batchSize));
    if (!engine)
        return 1;

    // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
    assert(engine->getNbBindings() == 2);
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims{engine->getBindingDimensions(i)};
        
        size_t size = accumulate(dims.d+1, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], batchSize * size * sizeof(float));

        // Resize CPU buffers to fit Tensor.
        if (engine->bindingIsInput(i))
            inputTensor.resize(size);
        else
            outputTensor.resize(size);
    }

    // Read input tensor from ONNX file.
    if (readTensor(inputFiles, inputTensor) != inputTensor.size())
    {
        cout << "Couldn't read input Tensor" << endl;
        return 1;
    }

    // Create Execution Context.
    context.reset(engine->createExecutionContext());

    Dims dims_i{engine->getBindingDimensions(0)};
    Dims4 inputDims{batchSize, dims_i.d[1], dims_i.d[2], dims_i.d[3]};
    context->setBindingDimensions(0, inputDims);

    doInference(context.get(), stream, inputTensor, outputTensor, bindings, batchSize);

    vector<string> referenceFiles;
    for (string path : inputFiles)
        referenceFiles.push_back(path.replace(path.rfind("input"), 5, "output"));
       
    // Try to read and compare against reference tensor from protobuf file.
    referenceTensor.resize(outputTensor.size());
    if (readTensor(referenceFiles, referenceTensor) != referenceTensor.size())
    {
        cout << "Couldn't read reference Tensor" << endl;
        return 1;
    }
    
    Dims dims_o{engine->getBindingDimensions(1)};
    int size = batchSize * dims_o.d[2] * dims_o.d[3];

    verifyOutput(outputTensor, referenceTensor, size);

    for (void* ptr : bindings)
        cudaFree(ptr);

    return 0;
}
