//
// Created by pavel on 25.04.20.
//

#ifndef CUDA_ADD_HISTOGRAMEXTRACTOR_H
#define CUDA_ADD_HISTOGRAMEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "ProcessedImage.h"

class HistogramExtractor {
private:
    void extractHistograms(int* histograms, int histogramCount, unsigned char* imputImg, int width, int height);
public:
    std::shared_ptr<ProcessedImage> ProcessImage(cv::Mat image);

};


#endif //CUDA_ADD_HISTOGRAMEXTRACTOR_H
