#include "add.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define RECT_SIZE 32
#define HIST_SIZE 9*256

__device__ bool isPixelLighterThanCentre(int y, int x, unsigned char centerVal, unsigned char* inImg, int width, int height) {
    if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
        return false;
    }
    int idx = x + y * gridDim.x;
    return inImg[idx] >= centerVal;
}

__device__ unsigned char getLBPVal(int x, int y, unsigned char* inImg, int width, int height) {
    unsigned char result = 0;
    int idx = x + y * gridDim.x;
    unsigned char center = inImg[idx];

    result |= ((unsigned char) isPixelLighterThanCentre(y-1, x-1, center, inImg, width, height)) << 7;
    result |= ((unsigned char) isPixelLighterThanCentre(y-1, x, center,    inImg, width, height)) << 6;
    result |= ((unsigned char) isPixelLighterThanCentre(y-1, x+1, center, inImg, width, height)) << 5;
    result |= ((unsigned char) isPixelLighterThanCentre(y, x+1, center, inImg, width, height)) << 4;
    result |= ((unsigned char) isPixelLighterThanCentre(y+1, x+1, center, inImg, width, height)) << 3;
    result |= ((unsigned char) isPixelLighterThanCentre(y+1, x,   center, inImg, width, height)) << 2;
    result |= ((unsigned char) isPixelLighterThanCentre(y+1, x-1, center, inImg, width, height)) << 1;
    result |= ((unsigned char) isPixelLighterThanCentre(y, x-1, center, inImg, width, height)) << 0;

    return result;

}

__global__ void lbpCUDA(unsigned char* inImg, unsigned char* outImg, int width, int height) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int idx = x + y * gridDim.x;

    outImg[idx] = getLBPVal(x, y, inImg, width, height);
}



__device__ bool getRectCoords(int &xFrom, int&yFrom, int startX, int startY, int width, int height) {
    int areaX, areaY;
    areaX = startX + blockIdx.x;
    areaY = startY;

    while (areaX > width - 256 * 3) {
        areaY++;
        areaX -= 256 * 3;

        if (areaY > height - 256 * 3) {
            xFrom = -1;
            yFrom = -1;
            return false;
        }
    }

    switch (threadIdx.x) {
        case 0:
            xFrom = areaX;
            yFrom = areaY;
            break;
        case 1:
            xFrom = areaX + 256;
            yFrom = areaY;
            break;
        case 2:
            xFrom = areaX + 2 * 256;
            yFrom = areaY;
            break;
        case 3:
            xFrom = areaX;
            yFrom = areaY + 256;
            break;
        case 4:
            xFrom = areaX + 256;
            yFrom = areaY + 256;
            break;
        case 5:
            xFrom = areaX + 2 * 256;
            yFrom = areaY + 256;
            break;
        case 6:
            xFrom = areaX;
            yFrom = areaY + 2 * 256;
            break;
        case 7:
            xFrom = areaX + 256;
            yFrom = areaY + 2 * 256;
            break;
        case 8:
            xFrom = areaX + 2 * 256;
            yFrom = areaY + 2 * 256;
            break;
    }

    return true;
}

__global__ void calculateHistograms(unsigned char* lbpImg, int* histogram, int startX, int startY, int width, int height) {
    int histStart = 9 * 256 * blockIdx.x + 256 * threadIdx.x;
    int xFrom, yFrom;
    getRectCoords(xFrom, yFrom, startX, startY, width, height);

    for (int i = 0; i < 256; i++) {
        histogram[histStart + i] = 0;
    }

    for (int x = 0; x < RECT_SIZE; x++) {
        for (int y = 0; y < RECT_SIZE; y++) {
            int value = lbpImg[x+y*width];
            histogram[value + histStart]++;
        }
    }
}




__global__ void calcuateDistances(int* histograms, int*dataset, double* distances, int datasetSize, int histogramSize, int histOrder) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadPos < histogramSize * datasetSize) {
        int histogramPos = threadPos / datasetSize;
        int datasetPos = threadPos % datasetSize;
        //double distance = 0;
        distances[threadPos] = 0;

        for (int i = 0; i < 9 * 256; i++) {
            distances[threadPos] += pow(
                    (double) (histograms[i + histogramPos * HIST_SIZE]) - dataset[i + datasetPos * HIST_SIZE], 2);
        }
        distances[threadPos] = sqrt(distances[threadPos]);
    }
}

__global__ void getKNNDistance(int neighbourCount, double*histogramDistances, double*knnDistances, int datasetSize, int histogramSize) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadPos < datasetSize * histogramSize) {
        double* nearestNeighbours = new double[neighbourCount];
        for (int i = 0; i < neighbourCount; i++){
            nearestNeighbours[i] = -1;
        }

        for (int i = 0; i < datasetSize; i++) {

            for (int j = 0; j < neighbourCount - 1; j++){
                if (j == 0) {
                    double dst = histogramDistances[i + threadPos * datasetSize]
                    if (nearestNeighbours[j] == -1 || nearestNeighbours[j] > dst) {
                        nearestNeighbours[j] = dst;
                    } else {
                        break;
                    }
                } else {
                    if (nearestNeighbours[j + 1] == -1 || nearestNeighbours[j + 1] > nearestNeighbours[j]) {
                        //switch values
                        double tmp = nearestNeighbours[j + 1];
                        nearestNeighbours[j + 1] = nearestNeighbours[j];
                        nearestNeighbours[j] = tmp;
                    } else {
                        break;
                    }
                }
            }
        }
        knnDistances[threadPos] = 0;
        for (int i = 0; i < neighbourCount; i++){
            knnDistances[threadPos] += nearestNeighbours[i];
        }
    }
}


void convertImageToLBP(unsigned char* imputImg, int width, int height, int* dataset) {
    unsigned char* Dev_InImg = nullptr;
    unsigned char* Dev_OutImg = nullptr;

    cudaMalloc((void**)&Dev_InImg,  height*width);
    cudaMalloc((void**)&Dev_OutImg, height*width);

    cudaMemcpy(Dev_InImg, imputImg, width * height, cudaMemcpyHostToDevice);

    dim3 gridImg(width, height);
    lbpCUDA<<<gridImg, 1>>>(Dev_InImg, Dev_OutImg, width, height);

    cudaMemcpy(imputImg, Dev_OutImg, width * height, cudaMemcpyDeviceToHost);

    cudaFree(Dev_InImg);

    int histogramSize = 9 * 256;        //VELIKOST HISTOGRAMU
    int histogramCount = (width - (histogramSize - 1)) * (height - (histogramSize - 1));  //POCET HISTOGRAMU V OBRAZKU
    int* histograms = new int[histogramCount * histogramSize];   //PAMET PRO HISTOGRAMY V POCITACI
    int histGrid = 1000000;            // POCET HISTOGRAMU, KTERE SE BUDOU ZAROVEN POCITAT NA GPU

    int* Dev_histograms = nullptr;
    cudaMalloc((void**)&Dev_histograms, histogramSize * histGrid * sizeof(int));

    dim3 gridHist(histGrid, 1);
    dim3 blockHist(9, 1,1); //HISTOGRAM JE SLOZENY Z 9 SUBHISTOGRAMU

    //VYPOCITEJ HISTOGRAMY A ULOZ JE V PAMETI
    int i = 0;
    int* writeFront = histograms;
    while (histGrid * (i + 1) <= histogramCount) {
        //startX = (histGrid * i) % width;
        //startY = (histGrid * i) / width;
        calculateHistograms<<<gridHist, blockHist>>>(Dev_OutImg, Dev_histograms, (histGrid * i) % width, (histGrid * i) / width, width, height);

        cudaMemcpy(writeFront, Dev_histograms, histogramSize * histogramCount * sizeof(int), cudaMemcpyDeviceToHost);
        writeFront += histGrid * histogramSize;
        i++;
    }
    int restHistGrid = histogramCount - histGrid * i;   //ZBYTEK NEDOPOCITANYCH
    if (restHistGrid > 0) {
        dim3 restGridHist(restHistGrid, 1);
        calculateHistograms<<<gridHist, blockHist>>>(Dev_OutImg, Dev_histograms, (histGrid * i) % width, (histGrid * i) / width, width, height);
        cudaMemcpy(writeFront, Dev_histograms, histogramSize * histGrid * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaFree(Dev_histograms);
    cudaFree(Dev_OutImg);

    //NAHRAJ DATASET DO GPU
    int* Dev_dataset = nullptr;
    cudaMalloc((void**)&Dev_dataset, 5000 * HIST_SIZE * sizeof(int));   //ALOKACE PAMETI PRO DATASET
    cudaMemcpy(Dev_dataset, dataset, 5000 * HIST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int biteSize = 1000;
    histGrid = (int)ceil((biteSize * HIST_SIZE) / 1024);

    double* Dev_distances = nullptr;
    cudaMalloc((void**)&Dev_distances, histGrid * 5000 * sizeof(double));   // ALOKACE PAMETI PRO VZDALENOSTI

    double* Dev_knnDistances = nullptr;
    cudaMalloc((void**)&Dev_knnDistances, histGrid * sizeof(double));   // ALOKACE PAMETI PRO K-NN VZDALENOSTI




    cudaFree(Dev_distances);
    cudaFree(Dev_knnDistances);
    cudaFree(Dev_dataset);
    delete [] histograms;
}
