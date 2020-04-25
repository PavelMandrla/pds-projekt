//
// Created by pavel on 25.04.20.
//

#include "HistogramExtractor.h"
#include "definitions.h"



std::shared_ptr<ProcessedImage> HistogramExtractor::ProcessImage(cv::Mat image) {
    ProcessedImage* result = new ProcessedImage();
    result->histogramCount = (image.cols - (AREA_SIZE - 1)) * (image.rows - (AREA_SIZE - 1));  //POCET HISTOGRAMU V OBRAZKU
    result->histograms = new int[result->histogramCount * HIST_SIZE];
    this->extractHistograms(result->histograms, result->histogramCount, image.data, image.cols, image.rows);

    return std::shared_ptr<ProcessedImage>(result);
}
