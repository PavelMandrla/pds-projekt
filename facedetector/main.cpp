#include <opencv2/opencv.hpp>
#include "add.h"
#include "HistogramExtractor.h"
#include "ProcessedImage.h"



using namespace cv;
using namespace std;

void loadDataset(int* setArr, string filename) {
    setArr = new int[5000*256*9];
    ifstream inFile;
    inFile.open(filename);
    string line;
    int i = 0;
    while (getline(inFile, line)) {
        stringstream ss(line);
        string value;
        while (getline(ss, value)) {
            setArr[i] = atoi(value.c_str());
            i++;
        }
    }
    inFile.close();
}

int main() {
    Mat Input_Image = imread("../swayze.png");
    cv::Mat bwImg;

    int* faces = nullptr;
    loadDataset(faces, "../dataset.csv");

    cv::cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);

    cout << "Height: " << bwImg.rows << ", Width: " << bwImg.cols << ", Channels: " << bwImg.channels() << endl;

    auto extractor = new HistogramExtractor();
    auto a = extractor->ProcessImage(bwImg);

    ofstream outfile("out.txt");

    for (int i = 0; i < a->histogramCount*9*255; i += 32) {
        outfile << a->histograms[i] << ",";


    }
    outfile << endl;

    outfile.close();

    //convertImageToLBP(bwImg.data, bwImg.cols, bwImg.rows, faces);

    //imwrite("../output.png", bwImg);

    delete [] faces;

    return 0;
}
