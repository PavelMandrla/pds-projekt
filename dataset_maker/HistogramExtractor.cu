//
// Created by pavel on 25.04.20.
//

#include "HistogramExtractor.h"
#include "definitions.h"

__device__ bool CUDA_isPixelLighterThanCentre(int y, int x, unsigned char centerVal, unsigned char* inImg, int width, int height) {
    if (x < 0 || y < 0 || x > width - 1 || y > height - 1) {
        return false;
    }
    int idx = x + y * gridDim.x;
    return inImg[idx] >= centerVal;
}

__device__ unsigned char CUDA_getLBPVal(int x, int y, unsigned char* inImg, int width, int height) {
    unsigned char result = 0;
    int idx = x + y * gridDim.x;
    unsigned char center = inImg[idx];

    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y - 1, x - 1, center, inImg, width, height)) << 7;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y - 1, x, center, inImg, width, height)) << 6;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y - 1, x + 1, center, inImg, width, height)) << 5;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y, x + 1, center, inImg, width, height)) << 4;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y + 1, x + 1, center, inImg, width, height)) << 3;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y + 1, x, center, inImg, width, height)) << 2;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y + 1, x - 1, center, inImg, width, height)) << 1;
    result |= ((unsigned char) CUDA_isPixelLighterThanCentre(y, x - 1, center, inImg, width, height)) << 0;

    return result;

}

__global__ void CUDA_calculateLBP(unsigned char* inImg, unsigned char* outImg, int width, int height) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int idx = x + y * gridDim.x;

    outImg[idx] = CUDA_getLBPVal(x, y, inImg, width, height);
}


__device__ bool getRectCoords(int &xFrom, int &yFrom, int startX, int startY, int width, int height) {
    int areaX, areaY;
    areaX = startX + blockIdx.x;
    areaY = startY;

    while (areaX > width - AREA_SIZE) {
        areaY++;
        areaX -= (width - AREA_SIZE) - 1;

        if (areaY > height - AREA_SIZE) {
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
            xFrom = areaX + GRID_SIZE;
            yFrom = areaY;
            break;
        case 2:
            xFrom = areaX + 2 * GRID_SIZE;
            yFrom = areaY;
            break;
        case 3:
            xFrom = areaX;
            yFrom = areaY + GRID_SIZE;
            break;
        case 4:
            xFrom = areaX + GRID_SIZE;
            yFrom = areaY + GRID_SIZE;
            break;
        case 5:
            xFrom = areaX + 2 * GRID_SIZE;
            yFrom = areaY + GRID_SIZE;
            break;
        case 6:
            xFrom = areaX;
            yFrom = areaY + 2 * GRID_SIZE;
            break;
        case 7:
            xFrom = areaX + GRID_SIZE;
            yFrom = areaY + 2 * GRID_SIZE;
            break;
        case 8:
            xFrom = areaX + 2 * GRID_SIZE;
            yFrom = areaY + 2 * GRID_SIZE;
            break;
    }

    return true;
}

__global__ void calculateHistograms(unsigned char* lbpImg, short* histogram, int startX, int startY, int width, int height) {
    int histStart = HIST_SIZE * blockIdx.x + 256 * threadIdx.x;
    int xFrom, yFrom;
    getRectCoords(xFrom, yFrom, startX, startY, width, height);

    for (int i = 0; i < 256; i++) {
        histogram[histStart + i] = 0;
    }

    for (int x = 0; x < GRID_SIZE; x++) {
        for (int y = 0; y < GRID_SIZE; y++) {
            int nX = x + xFrom;
            int nY = y + yFrom;
            int value = lbpImg[nX+nY*width];    //TODO -> ???
            histogram[value + histStart]++;
        }
    }
}


void HistogramExtractor::extractHistograms(short* histograms, int histogramCount, unsigned char* imputImg, int width, int height) {
    unsigned char *Dev_InImg = nullptr;
    cudaMalloc((void **) &Dev_InImg, height * width);
    cudaMemcpy(Dev_InImg, imputImg, width * height, cudaMemcpyHostToDevice);    //COPY IMAGE TO DEVICE

    unsigned char *Dev_OutImg = nullptr;
    cudaMalloc((void **) &Dev_OutImg, height * width);                          //CREATE IMAGE CONTAINING LBP VALUES

    dim3 grid_lbpCount(width, height);
    CUDA_calculateLBP<<<grid_lbpCount, 1>>>(Dev_InImg, Dev_OutImg, width, height);      //CALCULATE LBP VALUES

    cudaFree(Dev_InImg);

    int biteSize = 50000;         //NUMBER OF HISTOGRAMS THAT WILL BE CALCULATED AT ONCE

    short* Dev_histograms = nullptr;
    cudaMalloc((void **) &Dev_histograms, HIST_SIZE * biteSize * sizeof(short));

    dim3 grid_histograms(biteSize, 1);
    dim3 block_histograms(9,1,1);

    int i = 0;
    short* writeFront = histograms;
    while ((i+1)*biteSize <= histogramCount) {
        calculateHistograms<<<grid_histograms, block_histograms>>>(Dev_OutImg, Dev_histograms, (biteSize * i) % (width-95), (biteSize * i) / (width-95), width, height);
        cudaMemcpy(writeFront, Dev_histograms, HIST_SIZE * biteSize * sizeof(short), cudaMemcpyDeviceToHost);
        writeFront += HIST_SIZE * biteSize;
        i++;
    }
    int restSize = histogramCount - biteSize * i;
    if (restSize > 0) {
        dim3 grid_histogramsRest(restSize, 1);
        calculateHistograms<<<grid_histogramsRest, block_histograms>>>(Dev_OutImg, Dev_histograms, (biteSize * i) % (width-95), (biteSize * i) / (width - 95), width, height);
        cudaMemcpy(writeFront, Dev_histograms, HIST_SIZE * restSize * sizeof(short), cudaMemcpyDeviceToHost);
    }
    cudaFree(Dev_histograms);
    cudaFree(Dev_OutImg);
}

