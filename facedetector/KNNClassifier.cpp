//
// Created by pavel on 25.04.20.
//

#include "KNNClassifier.h"

KNNClassifier::KNNClassifier(int K, double threshold, short* dataset) {
    this->K = K;
    this->threshold = threshold;
    this->dataset = dataset;
}

std::list<cv::Rect> KNNClassifier::getFaces(std::shared_ptr<ProcessedImage> img) {
    std::list<cv::Rect> result;
    float* distances = new float[img->histogramCount];

    this->CalculateDistances(this->K, distances, this->dataset, 100, img->histograms, img->histogramCount);

    for (int i = 0; i < 1000; i++) {
        std::cout << distances[i] << ",";
    }
    std::cout << std::endl;


    return result;
}


