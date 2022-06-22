#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../InferenceEngine.h"


// Ce que j'ai fait: j'ai placé le fichier FindTensorRT.cmake dans: C:/Program Files/CMake/share/cmake3.20/Modules
// Faire pareil avec OpenCV!
// https://stackoverflow.com/questions/12573816/what-is-an-undefined-reference-unresolved-external-symbol-error-and-how-do-i-fix


int main() {
    std::cout << "Orbis, te saluto!" << std::endl;
    cv::Mat img = cv::imread("/home/feral/CLionProjects/TRT/media/ferret.jpg");
    int deviceCount;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    bool success;
    success = (e==cudaSuccess);
    InferenceEngine *nt;
    nt = new InferenceEngine();
    nt->doInference(img);
    delete nt;
    return 0;
}
