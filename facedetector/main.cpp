#include <opencv2/opencv.hpp>
#include "add.h"
#include "HistogramExtractor.h"
#include "KNNClassifier.h"
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
    KNNClassifier classifier(2, 15, faces);
    classifier.getFaces(extractor->ProcessImage(bwImg));

    //convertImageToLBP(bwImg.data, bwImg.cols, bwImg.rows, faces);

    //imwrite("../output.png", bwImg);

    delete [] faces;

    return 0;
}
