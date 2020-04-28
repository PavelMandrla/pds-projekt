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

int main(int argc, char** argv) {

    Mat Input_Image = imread(argv[1]);
    cv::Mat bwImg;

    int K = atoi(argv[2]);
    int threshold = atoi(argv[3]);

    short* dataset = new short[5000 * HIST_SIZE];
    loadDataset(dataset, "../dataset.csv");
    cv::cvtColor(Input_Image, bwImg, cv::COLOR_BGR2GRAY);


    auto extractor = new HistogramExtractor();
    KNNClassifier classifier(K, threshold, dataset);
    auto processed = extractor->ProcessImage(bwImg);
    auto faces = classifier.getFaces(processed);
    while (!faces.empty()) {
        int i = faces.front();
        faces.pop_front();

        int x = i % (bwImg.cols - 95);
        int y = i / (bwImg.cols - 95);
        cv::Rect rect(x, y, 96,96);
        rectangle(Input_Image, rect, Scalar(255,0,0), 1);
    }

    imwrite("myImageWithRect.png",Input_Image);

    delete [] dataset;

    return 0;
}
