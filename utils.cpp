//
// Created by Flavi on 15/10/2021.
//

#include <cuda_runtime_api.h>
#include "utils.h"

/**
 *
 * @param img
 * @param input_w
 * @param input_h
 * @return
 */
cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols * 1.0);  // ratio width
    float r_h = input_h / (img.rows * 1.0);  // ratio height
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    auto start = std::chrono::system_clock::now();
    cv::Mat re(h, w, CV_8UC3);
    auto end = std::chrono::system_clock::now();
    std::cout << "MatCPU: " <<std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000.f << "ms" << std::endl;

    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

cv::cuda::GpuMat preprocess_gpu(cv::Mat &img) {
    // todo: deux objectifs, 1. utiliser la mémoire Pinned pour diminuer le temps de transfert entre le CPU et le GPU
    // todo: 2. trouver un moyen de diminuer le temps de création des gpumat
    // cudaHostAlloc()
    // 365µs
    auto start = std::chrono::system_clock::now();
    cv::cuda::GpuMat d_out(YOLO_HEIGHT, YOLO_WIDTH, CV_8UC3, cv::Scalar(128, 128, 128));  // 640 640
    cv::cuda::GpuMat d_mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);  // 720 1280
    cv::cuda::GpuMat d_cvt(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
    cv::cuda::GpuMat d_gray(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1);
    cv::cuda::GpuMat d_resized(H_RESIZE, YOLO_WIDTH, CV_8UC3);  // 360 640
    auto end = std::chrono::system_clock::now();
    std::cout << "création GpuMat: " <<std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "µs" << std::endl;
    d_mat.upload(img);
    cv::cuda::cvtColor(d_mat, d_cvt, cv::COLOR_BGR2RGB);  // 150µs
    cv::cuda::cvtColor(d_cvt, d_gray, cv::COLOR_RGB2GRAY);
    cv::cuda::resize(d_mat, d_resized, SIZE, 0, 0, cv::INTER_LINEAR);  // 25µs
    d_resized.copyTo(d_out(cv::Rect(0, Y_RESIZE, YOLO_WIDTH, H_RESIZE)));
    void *data;  // ~100µs
    cudaMalloc(&data, BYTES);
    cudaMemcpyAsync(&data, d_resized.ptr(), BYTES, cudaMemcpyDeviceToDevice);

    return d_out;
}

int x_axisSnout(cv::Rect& r) {
    int x = (r.x + r.width / 2);
    return x;
}

int y_axisSnout(cv::Rect& r) {
    int y = (r.y + r.height / 2);
    return y;
}


cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r - l, b - t);
}


float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
            (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
            (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
            (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
            (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}


bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.conf > b.conf;
}


void nms(std::vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh, size_t size) {
    int n_iter = size / 4;
    int preDet_size = sizeof(Yolo::preDetection) / sizeof(float);
    std::map<float, std::vector<Yolo::Detection>> m;
    for (int i = 0; i < n_iter; i++) {
        Yolo::preDetection preDet;
        if (output[preDet_size * i + 4] <= conf_thresh) continue;
        memcpy(&preDet, &output[preDet_size * i], preDet_size * sizeof(float));
        // 1) on multiplie les probs par classe avec obj.
        std::vector<float> probs_per_class(preDet.prob_per_class, preDet.prob_per_class + sizeof(preDet.prob_per_class) / sizeof (float ));
        std::transform(probs_per_class.begin(), probs_per_class.end(), probs_per_class.begin(),
                       std::bind(std::multiplies<>(), std::placeholders::_1, preDet.obj_conf));
        // todo: prefer a lambda to std::bind.
        auto _i = std::max_element(probs_per_class.begin(), probs_per_class.end());
        auto index = std::distance(probs_per_class.begin(), _i);  // position dans le vecteur = classe
        float conf = *_i;  // c'est la valeur.. pas l'index
        // Transfert des données de prédection à la détection.
        Yolo::Detection det;
        float y[4];
        *y = xywh2xyxy(preDet.bbox, y);
        for (int j(0); j < 4; ++j)
            *(det.bbox + j) = *(y + j);
        det.conf = conf;
        det.class_id = (float) index;
        // Si on a aucune occurence pour la classe, on crée une entrée avec un std::vector<Yolo::Detection>
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        m[det.class_id].push_back(det);
    }
    // debug

    for (auto & it : m) {
        auto& dets = it.second;
        // on trie selon la confidence.
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t i = 0; i < dets.size(); ++i) {
                auto& item = dets[i];
            res.push_back(item);  // res: vecteur vide en entrée.
            for (size_t n = i + 1; n < dets.size(); ++n) {
                // todo: computeIoU ou iou?
                float overlap = computeIoU(item.bbox, dets[n].bbox);
                if (overlap > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

float xywh2xyxy(const float *arr, float *y) {
    *y = *arr - *(arr + 2) / 2;
    *(y + 1) = *(arr + 1) - *(arr + 3) / 2;
    *(y + 2) = *(arr + 0) + *(arr + 2) / 2;
    *(y + 3) = *(arr + 1) + *(arr + 3) / 2;
    return *y;
}

float xyxy2xywh(const float *arr, float *y) {
    *y = (*arr + *(arr + 2)) / 2;
    *(y + 1) = (*(arr + 1) + *(arr + 3)) / 2;
    *(y + 2) = *(arr + 2) - *arr;
    *(y + 3) = *(arr + 3) - *(arr + 1);
    return *y;
}

float computeIoU(float* bbox1, float* bbox2) {
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
        if (x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }

        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };

    auto IoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };
    return IoU(bbox1, bbox2);
}

int scaleCoords(cv::Mat &input, float *y, cv::Mat &resized) {
    float gain = std::min(resized.cols / input.cols, resized.rows / input.rows);
    // float pad =
    return 0;
}



