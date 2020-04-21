#include <opencv2/opencv.hpp>
#include "add.h"

using namespace cv;
using namespace std;


int main() {
    Mat Input_Image = imread("../swayze.png");
    cv::Mat bwImg;

    cv::cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);

    cout << "Height: " << bwImg.rows << ", Width: " << bwImg.cols << ", Channels: " << bwImg.channels() << endl;

    //Image_Inversion_CUDA(bwImg.data, bwImg.rows, bwImg.cols, bwImg.channels());
    convertImageToLBP(bwImg.data, bwImg.cols, bwImg.rows);

    imwrite("../output.png", bwImg);

    return 0;
}
