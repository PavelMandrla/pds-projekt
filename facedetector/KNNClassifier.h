//
// Created by pavel on 25.04.20.
//

#ifndef CUDA_ADD_KNNCLASSIFIER_H
#define CUDA_ADD_KNNCLASSIFIER_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include "ProcessedImage.h"

class KNNClassifier {
private:
    int K;
    double threshold;
    short* dataset;

    void CalculateDistances(int K, float* distances, short* dataset, int datasetSize, short* histograms, int histogramsCount);
public:
    KNNClassifier(int K, double threshold, short* dataset);

    std::list<cv::Rect> getFaces(std::shared_ptr<ProcessedImage> img);

};


#endif //CUDA_ADD_KNNCLASSIFIER_H
