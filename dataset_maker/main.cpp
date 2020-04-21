#include <stdio.h>
#include <dirent.h>
#include <string>
#include <vector>

#include <sstream>
#include <iomanip>
#include <opencv2/opencv.hpp>

#include "add.h"

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
        cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);
        resize(bwImg, bwImg, bwImg.size(), 96, 96, INTER_CUBIC);

        int* histogram = new int[9*256];
        convertImageToLBP(bwImg.data, bwImg.cols, bwImg.rows, histogram);

        for (int j = 0; j < (9*256) - 1; j++) {
            myfile << histogram[j] << ",";
        }
        myfile << histogram[(9*256) - 1] << endl;

    }
    myfile.close();

    return 0;
}