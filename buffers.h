//
// Created by Flavien on 08/07/2021.
//

#ifndef TENSORRTTEST_BUFFERS_H
#define TENSORRTTEST_BUFFERS_H

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <fstream>
#include <iterator>
#include <new>
#include <memory>
#include <cassert>


inline int64_t volume(const nvinfer1::Dims& d);

inline uint32_t getElementSize(nvinfer1::DataType t) noexcept;


template<typename AllocFunctor, typename FreeFunctor>
class GenericBuffer {
public:
    /**
     * Default type to FLOAT32
     * @param type
     */
    explicit GenericBuffer(nvinfer1::DataType type=nvinfer1::DataType::kFLOAT);
    GenericBuffer(size_t size, nvinfer1::DataType type);
    explicit GenericBuffer(GenericBuffer&& buffer);
    GenericBuffer& operator=(GenericBuffer&& buf);
    void* data();
    const void* data() const;
    size_t size() const;
    size_t nBytes() const;
    void resize(size_t newSize);
    void resize(const nvinfer1::Dims& dims);
    ~GenericBuffer();
private:
    nvinfer1::DataType m_type;
    size_t m_size{0};
    size_t m_capacity{0};
    void* m_buffer;
    AllocFunctor m_allocFunctor;
    FreeFunctor m_freeFunctor;
};

/**
 * Allocation functor for device.
 */
class DeviceAllocator {
public:
    bool operator()(void** ptr, size_t size) const;
};


/**
 * Free functor for device.
 */
class DeviceFree {
public:
    void operator()(void* ptr) const;
};

/**
 * Allocation functor for Host.
 */
class HostAllocator {
public:
    bool operator()(void** ptr, size_t size) const;
};


/**
 * Free functor for Host.
 */
class HostFree {
public:
    void operator()(void* ptr) const;
};


using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

/**
 * Regroupe les buffers CPU GPU
 */
class ManagedBuffer {
public:
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;
};


class BufferManager {
public:
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0);
    // A Supprimer
    BufferManager();

    /**
     *
     * @param engine
     * @param context
     */
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const nvinfer1::IExecutionContext* context=nullptr);
    std::vector<void*>& getDeviceBindings();
    const std::vector<void*>& getDeviceBindings() const;

    /**
     * On obtient le contenu du buffer sur le GPU
     * @param tensorName
     * @return
     */
    void* getDeviceBuffer(const std::string& tensorName) const;

    /**
     * On obtient le contenu du buffer sur le CPU
     * @param tensorName
     * @return
     */
    void* getHostBuffer(const std::string& tensorName) const;

    /**
     * ? je ne me souviens plus de quelle taille on parle. Interessant de pouvoir avoir la taille du buffer en sortie.
     * @param tensorName
     * @return
     */
    size_t size(const std::string& tensorName) const;
    // void dumpBuffer(std::ostream& os, const std::string& tensorName);
    // template <typename T>
    // void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount);

    /**
     * On transfère les données vers le GPU
     */
    void copyInputToDevice();

    /**
     * Vers le CPU depuis le GPU
     */
    void copyOutputToHost();
    void copyOutputToHostAsync(const cudaStream_t& stream = nullptr);
    // A Supprimer
    void setCudaEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine);
    void createBuffer();
    ~BufferManager() = default;
private:
    void* getBuffer(const bool isHost, const std::string& tensorName) const;
    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream=nullptr);
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    int mBatchSize;
    std::vector<std::unique_ptr<ManagedBuffer>> m_managedBuffers;
    std::vector<void*> m_deviceBindings;
};


#endif //TENSORRTTEST_BUFFERS_H
