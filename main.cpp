//
// Created by feral on 11/10/2021.
//

#include "Testing/VideoStream.h"
#include "InferenceEngine.h"
#include <cstdio>
#include <queue>
#include <vector>

/**
 * Video à analyser: /home/feral/Videos/road_traffic.mp4 /home/feral/Videos/video/film_dlc_02.wmv
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char* argv[]) {
    // argc = 3
    // argv[1] = video à analyser
    // argv[2] = chemin de sauvegarde.
    if (argc != 3)
        std::cerr << "Mauvais nombre d'arguments: " << argc << " Attendus: 3." << std::endl;
    std::string pathVideo = std::string(argv[1]);  // "/home/feral/Videos/video/film_dlc_02.wmv";
    std::string pathOutput = std::string(argv[2]);  // "/home/feral/CLionProjects/TRT/media/video_out.wmv";
    auto stream = VideoStream(pathVideo, pathOutput);
    auto inferenceEngine = InferenceEngine();

    std::queue<cv::Mat> foo;

    // FFMPEG
    // maxrate:v 10M
    // -loglevel debug
    // -y overwrite output files without asking
    // -framerate 25
    std::string commandFfmpeg = "ffmpeg  -y -f rawvideo -vcodec rawvideo -framerate 18 -pix_fmt bgr24 -s 1280x720 -i - -c:v h264_nvenc -r 18 /home/feral/Desktop/output.mkv";
    FILE* pipeout = popen(commandFfmpeg.data(), "w");
    std::chrono::time_point start = std::chrono::system_clock::now();
    std::chrono::time_point start1 = std::chrono::system_clock::now();
    std::chrono::time_point start2 = std::chrono::system_clock::now();
    std::chrono::time_point end1 = std::chrono::system_clock::now();
    std::chrono::time_point end2 = std::chrono::system_clock::now();
    std::vector<double> inference_time;
    std::vector<double> preprocessing_cpu_time;
    std::vector<double> preprocessing_gpu_time;
    int i(0);
    while (stream.getRemainingFrames() > 0) {
        if (!foo.empty())
            foo.pop();
        cv::Mat frame = stream.sendFrame();
        if (i == 200)
         cv::imwrite("/home/feral/Desktop/naive_frame.jpg", frame);
        start1 = std::chrono::system_clock::now();
        preprocess_img(frame, 640, 640);
        end1 = std::chrono::system_clock::now();
        preprocessing_cpu_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000.0);
        start1 = std::chrono::system_clock::now();
        preprocess_gpu(frame);
        end1 = std::chrono::system_clock::now();
        preprocessing_gpu_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1000.0);
        start2 = std::chrono::system_clock::now();
        inferenceEngine.doInference(frame);
        end2 = std::chrono::system_clock::now();
        inference_time.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count() / 1000.0);
        foo.push(frame);
        if (i == 200)
            cv::imwrite("/home/feral/Desktop/infer_frame.jpg", frame);
        fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipeout);

        ++i;
    }
    fflush(pipeout);
    pclose(pipeout);
    auto end = std::chrono::system_clock::now();
    float total_time = (float) std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "N frames: " << i << std::endl;
    std::cout << "FPS: " << (float) i / (total_time / 1000) << std::endl;
    std::cout << "Total: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.f << "s" << std::endl;
    auto n = inference_time.size();
    double sum(0.0);
    for (auto &it: inference_time)
        sum += it;
    double mean = sum / (double) n;
    std::cout << "Mean inference time: " << mean << "ms" << std::endl;
    n = preprocessing_cpu_time.size();
    sum = 0.0;
    for (auto &it: preprocessing_cpu_time)
        sum += it;
    mean = sum / (double) n;
    std::cout << "Mean preprocessing CPU time: " << mean << "ms" << std::endl;
    n = preprocessing_gpu_time.size();
    sum = 0.0;
    for (auto &it: preprocessing_gpu_time)
        sum += it;
    mean = sum / (double) n;
    std::cout << "Mean preprocessing GPU time: " << mean << "ms" << std::endl;
    return 0;
}
