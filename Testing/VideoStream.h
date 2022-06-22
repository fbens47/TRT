//
// Created by feral on 11/10/2021.
//

#ifndef YOLO_VIDEOSTREAM_H
#define YOLO_VIDEOSTREAM_H

#include <opencv2/videoio.hpp>
#include <iostream>

/**
 * Créer un flux video qui envoie des frames à un réseau de neurones,
 * et qui reprend les frames analysées pour créer une nouvelle vidéo
 */
class VideoStream {
public:
    /**
     * Crée les stream entrée / sortie.
     * cv::VideoWriter videoWriter(path, cv::CAP_OPENCV_MJPEG, fourcc, fps, *size);
     * cv::VideoCapture videoCapture(path, cv::CAP_ANY);
     */
    VideoStream(std::string pathVideoIn, std::string pathVideoOut);
    ~VideoStream();
    // m_cap.grab()
    cv::Mat sendFrame();
    // m_writer.
    void writeFrame(const cv::Mat& frame);
    [[nodiscard]] int getRemainingFrames() const;
private:
    std::string m_pathVideoToAnalyse;
    std::string m_pathVideoToSave;
    cv::VideoCapture *m_cap;
    cv::VideoWriter *m_writer;
    int m_nFrameCounter;
};




#endif //YOLO_VIDEOSTREAM_H
