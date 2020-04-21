#include "add.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define RECT_SIZE 32

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



void convertImageToLBP(unsigned char* imputImg, int width, int height, int* histograms) {
    unsigned char* Dev_InImg = nullptr;
    unsigned char* Dev_OutImg = nullptr;

    cudaMalloc((void**)&Dev_InImg,  height*width);
    cudaMalloc((void**)&Dev_OutImg, height*width);

    cudaMemcpy(Dev_InImg, imputImg, width * height, cudaMemcpyHostToDevice);

    dim3 gridImg(width, height);
    lbpCUDA<<<gridImg, 1>>>(Dev_InImg, Dev_OutImg, width, height);

    cudaMemcpy(imputImg, Dev_OutImg, width * height, cudaMemcpyDeviceToHost);

    cudaFree(Dev_InImg);

    int histSize = 9*256;
    int histCount = 1;

    int* Dev_histograms = nullptr;
    cudaMalloc((void**)&Dev_histograms,  histSize * histCount * sizeof(int));

    dim3 gridHist(histCount, 1);
    dim3 blockHist(9, 1,1);
    calculateHistograms<<<gridHist, blockHist>>>(Dev_OutImg, Dev_histograms, 0, 0, width, height);

    cudaMemcpy(histograms, Dev_histograms,  histSize * histCount * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(Dev_histograms);
    cudaFree(Dev_OutImg);
}
