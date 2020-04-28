#include <stdio.h>
#include <dirent.h>
#include <string>
#include <vector>

#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "HistogramExtractor.h"
#include "ProcessedImage.h"
//#include "add.h"

using namespace cv;
using namespace std;

vector<int> getLBPH(Mat img) {

}

int main(int argc, const char**argv) {
    ofstream myfile;
    myfile.open("dataset.csv");

    for (int i = 0; i < 5000; i++) {
        std::stringstream ss;
        ss << "../faces/" << std::setw(5) << std::setfill('0') << i << ".png";
        std::string s = ss.str();

        Mat Input_Image = imread(ss.str());
        Mat bwImg;
        Mat resized;
        cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);
        resize(bwImg, resized, Size(96,96));

        HistogramExtractor extractor;
        auto processed = extractor.ProcessImage(resized);


        //int* histogram = new int[9*256];
        //convertImageToLBP(bwImg.data, bwImg.cols, bwImg.rows, histogram);

        for (int j = 0; j < (9*256) - 1; j++) {
            //myfile << histogram[j] << ",";
            myfile << processed->histograms[j] << ",";
        }
        //myfile << histogram[(9*256) - 1] << endl;
        myfile << processed->histograms[(9*256) - 1] << endl;

    }
    myfile.close();

/*
    Mat Input_Image = imread("../faces/00037.png");
    Mat bwImg;
    Mat resized;
    cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);
    resize(bwImg, resized, Size (96,96));

    imwrite("out.png", resized);
*/

    return 0;
}