//
// Created by feral on 11/10/2021.
//

#include "VideoStream.h"

VideoStream::VideoStream(std::string pathVideoIn, std::string pathVideoOut): m_pathVideoToAnalyse(pathVideoIn), m_pathVideoToSave(pathVideoOut) {
    m_cap = new cv::VideoCapture(m_pathVideoToAnalyse, cv::CAP_ANY);
    if (!m_cap->isOpened())
        std::cerr << "Erreur lors de l'ouverture du fichier";
    m_nFrameCounter = (int) m_cap->get(cv::CAP_PROP_FRAME_COUNT);
    m_writer = new cv::VideoWriter();
}

VideoStream::~VideoStream() {
    if (m_cap->isOpened())
        m_cap->release();
    delete m_cap;

    //
    delete m_writer;

}

void VideoStream::writeFrame(const cv::Mat &frame) {
    if (!m_writer->isOpened()) {
        int fourcc = m_cap->get(cv::CAP_PROP_FOURCC);
        double FPS = m_cap->get(cv::CAP_PROP_FPS);
        std::cout << "FPS = " << FPS;
        int height = m_cap->get(cv::CAP_PROP_FRAME_HEIGHT);
        int width = m_cap->get(cv::CAP_PROP_FRAME_WIDTH);
        m_writer->open(m_pathVideoToSave, fourcc, FPS, frame.size(), true);
    }
    m_writer->write(frame);
}

cv::Mat VideoStream::sendFrame() {
    cv::Mat out;
    if (!m_cap->read(out))
        std::cerr << "Image vide";
    --m_nFrameCounter;
    return out;
}

int VideoStream::getRemainingFrames() const {
    return m_nFrameCounter;
}
