//
// Created by Flavi on 09/07/2021.
//

#include "InferenceEngine.h"
#include <ostream>

#include <utility>

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
// OUTPUT BLOB NAME
// INPUT BLOB NAME voir Netron.

template<class T>
void InferDeleter::operator()(T* obj) const {
    if (obj) {
        obj->destroy();
    }
}

InferenceEngine::InferenceEngine() {
    m_stream = new cudaStream_t;
    CUDA_CHECK(cudaStreamCreate(m_stream));
    m_builder = nullptr;
    buildNetworkSerial();
    for (size_t i = 0; i < m_engine->getNbBindings(); ++i)
    {
        // Calcul de la taille que devra avoir le buffer.
        auto binding_size = getSizeByDim(m_engine->getBindingDimensions(i)) * sizeof(float);
        // Allocation de la mémoire sur GPU
        CUDA_CHECK(cudaMalloc(&m_buffers[i], binding_size));
        if (m_engine->bindingIsInput(i))
            inputSize = binding_size;

        else
            if (i == 1)
                outputSize397 = binding_size;
            else if (i == 2)
                outputSize458 = binding_size;
            else if (i == 3)
                outputSize519 = binding_size;
            else if (i == 4)
                outputSize561 = binding_size;
    }
    // std::cout << inputSize << "  " << outputSize561 << std::endl;
    data = new float [inputSize];
    prob = new float [outputSize561];
}

void InferenceEngine::buildNetwork() {
    // Possibles problèmes ici...
    Logger gLogger;
    m_builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = m_builder->createNetworkV2(explicitBatch); // args?
    auto parser = nvonnxparser::createParser(*network, gLogger);
    parser->parseFromFile(m_pathONNX.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    nvinfer1::IBuilderConfig *config = m_builder->createBuilderConfig();
    m_builder->setMaxBatchSize(1);
    // TODO: sérialiser. C'est sérieusement long de charger le réseau...
    if (m_builder->platformHasFastFp16())
    {
       std::cout << "kFP16";
       config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    m_engine = m_builder->buildEngineWithConfig(*network, *config);
    m_context = m_engine->createExecutionContext();
    m_builder->destroy();
}


void InferenceEngine::doInference(cv::Mat &inputFrame) {
    cv::Mat resizedImg;
    // auto size = cv::Size(INPUT_W, INPUT_H);

    resizedImg = preprocess_img(inputFrame, INPUT_W, INPUT_H);
    int i(0);
    for (int row(0); row < INPUT_H; ++row) {
        uchar* uc_pixel = resizedImg.data + row * resizedImg.step;
        for (int col(0); col < INPUT_W; ++col) {
            // ?
            *(data + i) = (float) uc_pixel[2] / 255.f;
            // *mean += *(data + i);
            *(data +i + INPUT_H * INPUT_W) = (float) uc_pixel[1] / 255.f;
            // *(mean + 1) += *(data +i + INPUT_H * INPUT_W);
            *(data + i + 2 * INPUT_H * INPUT_W) = (float)uc_pixel[1] / 255.f;
            uc_pixel += 3;
            ++i;
        }
    }
    auto start = std::chrono::system_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(m_buffers[0], data, inputSize, cudaMemcpyHostToDevice, *m_stream));
    auto end = std::chrono::system_clock::now();
    std::cout << "Host to device: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms" << std::endl;
    start = std::chrono::system_clock::now();
    m_context->enqueueV2(m_buffers, *m_stream, nullptr);
    end = std::chrono::system_clock::now();
    std::cout << "Inference: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms" << std::endl;
    // 4 correspond à 561; pour avoir les données de cette sortie
    start = std::chrono::system_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(prob, m_buffers[4], outputSize561, cudaMemcpyDeviceToHost, *m_stream));
    end = std::chrono::system_clock::now();
    std::cout << "Device to host: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.f << "ms" << std::endl;
    start = std::chrono::system_clock::now();
    cudaStreamSynchronize(*m_stream);
    std::vector<Yolo::Detection> detections;
    size_t nIter = outputSize561 / 8;

    nms(detections, prob, CONF_THRESH, NMS_THRESH, nIter);
    end = std::chrono::system_clock::now();
    std::cout << "NMS: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs" << std::endl;

    int counter_class_snout(0);
    int counter_class_eye(0);
    int counter_class_ear(0);
    for (auto &it: detections) {
        int class_id = (int) it.class_id;
        if (class_id == 0 && counter_class_snout == 0) {
            ++counter_class_snout;
            float y[4];
            *y = xyxy2xywh(it.bbox, y);
            auto r = get_rect(inputFrame, y);
            // Position du museau
            // angle gauche sup (x + width / 2) (y + height / 2)
            int x = x_axisSnout(r);
            int y_axis = y_axisSnout(r);
            cv::circle(inputFrame, cv::Point(x, y_axis), 20, cv::Scalar(0, 0, 255), 4, 8, 0);
            cv::rectangle(inputFrame, r, cv::Scalar(0, 255, 0), 5, 8, 0);

        }
        else if (class_id == 1 && counter_class_eye < 3) {
            float y[4];
            *y = xyxy2xywh(it.bbox, y);
            auto r = get_rect(inputFrame, y);
            cv::rectangle(inputFrame, r, cv::Scalar(0, 255, 0), 5, 8, 0);
            ++counter_class_eye;
        }
        else if (class_id == 2 && counter_class_ear < 3) {
            ++counter_class_ear;
            float y[4];
            *y = xyxy2xywh(it.bbox, y);
            auto r = get_rect(inputFrame, y);
            cv::rectangle(inputFrame, r, cv::Scalar(0, 255, 0), 5, 8, 0);
        }
    }
}

InferenceEngine::~InferenceEngine() {

    CUDA_CHECK(cudaStreamDestroy(*m_stream));
    for (int i(0); i < 5; ++i)
        CUDA_CHECK(cudaFree(m_buffers[i]));
    m_context->destroy();  // TRT_DEPRECATED, mais incapable d'utiliser le destructeur.
    // m_engine.release();
}

void* InferenceEngine::createBuffers() {
    void* buffers[5];
    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize397);
    cudaMalloc(&buffers[2], outputSize458);
    cudaMalloc(&buffers[3], outputSize519);
    cudaMalloc(&buffers[4], outputSize561);
    return buffers;
}

void InferenceEngine::buildNetworkSerial() {
    Logger gLogger;
    char *trtModelStream = nullptr;
    std::ifstream fin("/home/feral/CLionProjects/TRT/deep_net/last.bin", std::ios::binary);
    size_t size = 0;
    fin.seekg(0, fin.end);
    size = fin.tellg();
    fin.seekg(0, fin.beg);
    trtModelStream = new char[size];
    fin.read(trtModelStream, size);
    fin.close();
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    m_engine = runtime->deserializeCudaEngine(trtModelStream, size);
    m_context = m_engine->createExecutionContext();
    delete[] trtModelStream;
    delete runtime;
}


size_t getSizeByDim(const nvinfer1::Dims& dims){
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}
