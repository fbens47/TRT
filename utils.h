//
// Created by Flavi on 16/10/2021.
//

#ifndef TENSORRTTEST_UTILS_H
#define TENSORRTTEST_UTILS_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <vector>
#include <chrono>
#include <utility>
#include "yolo.hpp"

#define FRAME_WIDTH  1280
#define FRAME_HEIGHT 720
#define YOLO_WIDTH   640
#define YOLO_HEIGHT  640
#define RATIO_WIDTH  YOLO_WIDTH  / (FRAME_WIDTH  * 1.f)
#define RATIO_HEIGHT YOLO_HEIGHT / (FRAME_HEIGHT * 1.f)
#define H_RESIZE     RATIO_WIDTH * FRAME_HEIGHT
#define Y_RESIZE     (YOLO_HEIGHT - H_RESIZE) / 2
#define SIZE         cv::Size(YOLO_WIDTH, H_RESIZE)
#define BYTES        YOLO_WIDTH * YOLO_HEIGHT * sizeof(float)

/**
 * Ce que ça fait:
 * @param img
 * @param input_w
 * @param input_h
 * @return
 */
cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h);

cv::cuda::GpuMat preprocess_gpu(cv::Mat& img);

float xywh2xyxy(const float arr[4], float y[4]);

float xyxy2xywh(const float arr[4], float y[4]);

int scaleCoords(cv::Mat& input, float y[4], cv::Mat& resized);

cv::Rect get_rect(cv::Mat& img, float bbox[4]);

float iou(float lbox[4], float rbox[4]);

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b);

void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh, size_t size);

float computeIoU(float* bbox1, float* bbox2);

int x_axisSnout(cv::Rect& r);

int y_axisSnout(cv::Rect& r);
#endif //TENSORRTTEST_UTILS_H
