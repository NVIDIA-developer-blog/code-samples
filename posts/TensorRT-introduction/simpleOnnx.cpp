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
#include "cudaWrapper.h"
#include "ioHelper.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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
    unique_ptr<IBuilder, Destroy<IBuilder>> builder{createInferBuilder(gLogger)};
    unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{builder->createNetwork()};
    unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{nvonnxparser::createParser(*network, gLogger)};

    if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
    {
        cout << "ERROR: could not parse input engine." << endl;
        return nullptr;
    }

    // Build TensorRT engine optimized based on for batch size of input data provided.
    builder->setMaxBatchSize(batchSize);
    // Allow TensorRT to use fp16 mode kernels internally.
    // Note that Input and Output tensors will still use 32 bit float type by default.
    builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);

    return builder->buildCudaEngine(*network); // Build and return TensorRT engine.
}

ICudaEngine* getCudaEngine(string const& onnxModelPath, int batchSize)
{
    string enginePath{getBasename(onnxModelPath) + "_batch" + to_string(batchSize) + ".engine"};
    ICudaEngine* engine{nullptr};

    string buffer = readBuffer(enginePath);
    if (buffer.size())
    {
        // Try to deserialize engine.
        unique_ptr<IRuntime, Destroy<IRuntime>> runtime{createInferRuntime(gLogger)};
        engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
    }

    if (!engine)
    {
        // Fallback to creating engine from scratch.
        engine = createCudaEngine(onnxModelPath, batchSize);

        if (engine)
        {
            unique_ptr<IHostMemory, Destroy<IHostMemory>> engine_plan{engine->serialize()};
            // Try to save engine for future uses.
            writeBuffer(engine_plan->data(), engine_plan->size(), enginePath);
        }
    }
    return engine;
}

static int getBindingInputIndex(IExecutionContext* context)
{
    return !context->getEngine().bindingIsInput(0); // 0 (false) if bindingIsInput(0), 1 (true) otherwise
}

void launchInference(IExecutionContext* context, cudaStream_t stream, vector<float> const& inputTensor, vector<float>& outputTensor, void** bindings, int batchSize)
{
    int inputId = getBindingInputIndex(context);

    cudaMemcpyAsync(bindings[inputId], inputTensor.data(), inputTensor.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    context->enqueue(batchSize, bindings, stream, nullptr);
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

void softmax(vector<float>& tensor, int batchSize)
{
    size_t batchElements = tensor.size() / batchSize;

    for (int i = 0; i < batchSize; ++i)
    {
        float* batchVector = &tensor[i * batchElements];
        double maxValue = *max_element(batchVector, batchVector + batchElements);
        double expSum = accumulate(batchVector, batchVector + batchElements, 0.0, [=](double acc, float value) { return acc + exp(value - maxValue); });

        transform(batchVector, batchVector + batchElements, batchVector, [=](float input) { return static_cast<float>(std::exp(input - maxValue) / expSum); });
    }
}

void verifyOutput(vector<float> const& outputTensor, vector<float> const& referenceTensor)
{
    for (size_t i = 0; i < referenceTensor.size(); ++i)
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
    engine.reset(getCudaEngine(onnxModelPath, batchSize));
    if (!engine)
        return 1;

    // Assume networks takes exactly 1 input tensor and outputs 1 tensor.
    assert(engine->getNbBindings() == 2);
    assert(engine->bindingIsInput(0) ^ engine->bindingIsInput(1));

    for (int i = 0; i < engine->getNbBindings(); ++i)
    {
        Dims dims{engine->getBindingDimensions(i)};
        size_t size = accumulate(dims.d, dims.d + dims.nbDims, batchSize, multiplies<size_t>());
        // Create CUDA buffer for Tensor.
        cudaMalloc(&bindings[i], size * sizeof(float));

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

    // Apply a softmax on the CPU to create a normalized distribution suitable for measuring relative error in probabilities.
    softmax(outputTensor, batchSize);
    softmax(referenceTensor, batchSize);

    verifyOutput(outputTensor, referenceTensor);

    for (void* ptr : bindings)
        cudaFree(ptr);

    return 0;
}
