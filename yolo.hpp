//
// Created by Flavi on 15/10/2021.
//

#ifndef TENSORRTTEST_YOLO_HPP
#define TENSORRTTEST_YOLO_HPP

namespace Yolo {
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int NUM_CLASSES = 3;
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int LOCATIONS = 4;
    struct alignas(float) preDetection {
        float bbox[LOCATIONS];
        float obj_conf;
        float prob_per_class[NUM_CLASSES];
    };
    struct Detection {
        float bbox[LOCATIONS];
        float conf;
        float class_id;
    };
};

#endif //TENSORRTTEST_YOLO_HPP
