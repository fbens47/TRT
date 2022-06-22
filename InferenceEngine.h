//
// Created by Flavi on 09/07/2021.
//

#ifndef TENSORRTTEST_INFERENCEENGINE_H
#define TENSORRTTEST_INFERENCEENGINE_H


#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <iostream>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInferPluginUtils.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <fstream>
#include "cuda_utils.h"
#include "yolo.hpp"
#include "utils.h"
#include "buffers.h"

#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1


struct InferDeleter {
    template <class T>
    void operator()(T* obj) const;
};

template <typename T>
using UniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger: public nvinfer1::ILogger {
    void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept override {
        if (severity != nvinfer1::ILogger::Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};


class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    void buildNetwork();
    void buildNetworkSerial();
    void doInference(cv::Mat &inputFrame);

private:
    /**
     * à utiliser dans doInference pour créer le buffer.
     * @return
     */
    void* createBuffers();
    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IBuilder *m_builder;
    nvinfer1::IExecutionContext *m_context;
    cudaStream_t *m_stream;
    void* m_buffers[5];
    // ce sont les données d'entrées à envoyer vers le buffer device
    float *data;
    // ce sont les données de sortie qui reçoivent les données du buffer device
    float *prob;
    size_t outputSize397;  // 614400
    size_t outputSize458;  // 153600
    size_t outputSize519;  // 38400
    size_t outputSize561;  // 806400
    size_t inputSize;
    std::string m_pathONNX = "/home/feral/CLionProjects/TRT/deep_net/last.onnx";
};


size_t getSizeByDim(const nvinfer1::Dims& dims);

#endif //TENSORRTTEST_INFERENCEENGINE_H
