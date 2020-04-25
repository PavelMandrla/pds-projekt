//
// Created by pavel on 25.04.20.
//

#include "KNNClassifier.h"

KNNClassifier::KNNClassifier(int K, double threshold, int* dataset) {
    this->K = K;
    this->threshold = threshold;
    this->dataset = dataset;
}

std::list<cv::Rect> KNNClassifier::getFaces(std::shared_ptr<ProcessedImage> img) {
    std::list<cv::Rect> result;
    double* distances = new double[img->histogramCount];
    this->CalculateDistances(this->K, distances, this->dataset, 5000, img->histograms, img->histogramCount);

    return result;
}


