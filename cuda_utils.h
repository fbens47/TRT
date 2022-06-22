//
// Created by Flavi on 15/10/2021.
//

#ifndef TENSORRTTEST_CUDA_UTILS_H
#define TENSORRTTEST_CUDA_UTILS_H


#include <cuda_runtime_api.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

#endif //TENSORRTTEST_CUDA_UTILS_H
