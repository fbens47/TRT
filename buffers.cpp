//
// Created by Flavi on 08/07/2021.
//

#include "buffers.h"

int64_t volume(const nvinfer1::Dims &d) {
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

uint32_t getElementSize(nvinfer1::DataType t) noexcept {
    switch(t) {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    return 0;
}

template<typename AllocFunctor, typename FreeFunctor>
GenericBuffer<AllocFunctor, FreeFunctor>::GenericBuffer(nvinfer1::DataType type) {
    m_buffer = nullptr;
    m_capacity = 0;
    m_size = 0;
    m_type = type;
}

template<typename AllocFunctor, typename FreeFunctor>
GenericBuffer<AllocFunctor, FreeFunctor>::GenericBuffer(size_t size, nvinfer1::DataType type) {
    m_buffer = nullptr;
    m_capacity = 0;
    m_size = size;
    m_type = type;
    if (!m_allocFunctor(&m_buffer, this->nBytes()))
        throw std::bad_alloc();
}

template<typename AllocFunctor, typename FreeFunctor>
GenericBuffer<AllocFunctor, FreeFunctor>::GenericBuffer(GenericBuffer &&buffer) {
    m_size = buffer.m_size;
    m_capacity = buffer.m_capacity;
    m_type = buffer.m_type;
    m_buffer = buffer.m_buffer;
    buffer.m_size = 0;
    buffer.m_capacity = 0;
    buffer.m_type = nvinfer1::DataType::kFLOAT;
    buffer.m_buffer = nullptr;
}

template<typename AllocFunctor, typename FreeFunctor>
GenericBuffer<AllocFunctor, FreeFunctor>& GenericBuffer<AllocFunctor, FreeFunctor>::operator=(GenericBuffer &&buffer) {
    if (this != &buffer) {
        m_freeFunctor(m_buffer);
        m_size = buffer.m_size;
        m_capacity = buffer.m_capacity;
        m_type = buffer.m_type;
        m_buffer = buffer.m_buffer;
        buffer.m_size = 0;
        buffer.m_capacity = 0;
        buffer.m_buffer = nullptr;
    }
    return *this;
}

template<typename AllocFunctor, typename FreeFunctor>
void *GenericBuffer<AllocFunctor, FreeFunctor>::data() {
    return m_buffer;
}

template<typename AllocFunctor, typename FreeFunctor>
const void *GenericBuffer<AllocFunctor, FreeFunctor>::data() const {
    return m_buffer;
}

template<typename AllocFunctor, typename FreeFunctor>
size_t GenericBuffer<AllocFunctor, FreeFunctor>::size() const {
    return m_size;
}

template<typename AllocFunctor, typename FreeFunctor>
size_t GenericBuffer<AllocFunctor, FreeFunctor>::nBytes() const {
    return this->size() * getElementSize(m_type);;
}

template<typename AllocFunctor, typename FreeFunctor>
void GenericBuffer<AllocFunctor, FreeFunctor>::resize(size_t newSize) {
    m_size = newSize;
    if (m_capacity < newSize) {
        m_freeFunctor(m_buffer);
        if (!m_allocFunctor(&m_buffer, this->nBytes())) {
            throw std::bad_alloc();
        }
        m_capacity = newSize;
    }
}

template<typename AllocFunctor, typename FreeFunctor>
void GenericBuffer<AllocFunctor, FreeFunctor>::resize(const nvinfer1::Dims &dims) {
    return this->resize(volume(dims));
}

template<typename AllocFunctor, typename FreeFunctor>
GenericBuffer<AllocFunctor, FreeFunctor>::~GenericBuffer() {
    m_freeFunctor(m_buffer);
}


bool DeviceAllocator::operator()(void **ptr, size_t size) const {
    return cudaMalloc(ptr, size) == cudaSuccess;
}

void DeviceFree::operator()(void *ptr) const {
    cudaFree(ptr);
}

bool HostAllocator::operator()(void **ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
}

void HostFree::operator()(void *ptr) const {
    free(ptr);
}

BufferManager::BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                             const nvinfer1::IExecutionContext *context): mBatchSize(0), m_engine(engine) {
    // assert(m_engine->hasImplicitBatchDimension() )
    // Création des tampons Host et Device
    for (int i(0); i < m_engine->getNbBindings(); ++i) {
        auto dims = context ? context->getBindingDimensions(i) : m_engine->getBindingDimensions(i);
        size_t vol = context || !mBatchSize ? 1 : static_cast<size_t>(mBatchSize);
        nvinfer1::DataType type = m_engine->getBindingDataType(i);
        vol *= volume(dims);
        std::unique_ptr<ManagedBuffer> managedBuffer{new ManagedBuffer};
        managedBuffer->deviceBuffer = DeviceBuffer(vol, type);
        managedBuffer->hostBuffer = HostBuffer(vol, type);
        m_deviceBindings.emplace_back(managedBuffer->deviceBuffer.data());
        m_managedBuffers.emplace_back(std::move(managedBuffer));
    }

}

std::vector<void *> &BufferManager::getDeviceBindings() {
    return m_deviceBindings;
}

const std::vector<void *> &BufferManager::getDeviceBindings() const {
    return m_deviceBindings;
}

void *BufferManager::getBuffer(const bool isHost, const std::string &tensorName) const {
    int index = m_engine->getBindingIndex(tensorName.c_str());
    if (index == -1)
        return nullptr;
    return (isHost ? m_managedBuffers[index]->hostBuffer.data() : m_managedBuffers[index]->deviceBuffer.data());
}

void BufferManager::memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async,
                                  cudaStream_t const &stream) {
    for (int i(0); i < m_engine->getNbBindings(); ++i) {
        void* dstPtr = deviceToHost ? m_managedBuffers[i]->hostBuffer.data() : m_managedBuffers[i]->deviceBuffer.data();
        const void* srcPtr = deviceToHost ? m_managedBuffers[i]->deviceBuffer.data() : m_managedBuffers[i]->hostBuffer.data();
        const size_t byteSize = m_managedBuffers[i]->hostBuffer.nBytes();
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        if ((copyInput && m_engine->bindingIsInput(i)) || (!copyInput && !m_engine->bindingIsInput(i))) {
            if (async)
                cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream);
            else
                cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType);
        }
    }
}

void *BufferManager::getDeviceBuffer(const std::string &tensorName) const {
    return getBuffer(false, tensorName);
}

void *BufferManager::getHostBuffer(const std::string &tensorName) const {
    return getBuffer(true, tensorName);
}

void BufferManager::copyInputToDevice() {
    memcpyBuffers(true, false, false);
}

void BufferManager::copyOutputToHost() {
    memcpyBuffers(false, true, false);
}

void BufferManager::copyOutputToHostAsync(cudaStream_t const &stream) {
    memcpyBuffers(false, true, true, stream);
}

size_t BufferManager::size(const std::string &tensorName) const {
    int index = m_engine->getBindingIndex(tensorName.c_str());
    if (index == -1)
        return kINVALID_SIZE_VALUE;
    return m_managedBuffers[index]->hostBuffer.nBytes();
}

void BufferManager::setCudaEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine) {
    m_engine = engine;
}

BufferManager::BufferManager() {
    m_engine = nullptr;
}
