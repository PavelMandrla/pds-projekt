#include <opencv2/opencv.hpp>
#include "add.h"
#include "HistogramExtractor.h"
#include "KNNClassifier.h"
#include "ProcessedImage.h"
#include "definitions.h"


using namespace cv;
using namespace std;

void loadDataset(short* setArr, string filename) {
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

    short* faces = new short[5000*256*9];
    loadDataset(faces, "../dataset.csv");
    cv::cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);

    cout << "Height: " << bwImg.rows << ", Width: " << bwImg.cols << ", Channels: " << bwImg.channels() << endl;

    auto extractor = new HistogramExtractor();
    KNNClassifier classifier(3, 15, faces);
    auto processed = extractor->ProcessImage(bwImg);
    /*
    for (int i = 0; i < HIST_SIZE; i++) {
        cout << processed->histograms[i] << " - " << processed->histograms[i + HIST_SIZE] << " - " << processed->histograms[i + 2*HIST_SIZE] << " - " << processed->histograms[i + 3*HIST_SIZE] << " - " << processed->histograms[i + 4*HIST_SIZE] << endl;
    }
    */
    classifier.getFaces(processed);

    delete [] faces;

    return 0;
}
