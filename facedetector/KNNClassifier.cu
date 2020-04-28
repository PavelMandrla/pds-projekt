//
// Created by pavel on 25.04.20.
//

#include "KNNClassifier.h"
#include "definitions.h"


__global__ void CUDA_calcuateDistances(short* histograms, short*dataset, float* distances, int datasetSize, int histogramSize) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadPos < histogramSize * datasetSize) {
        int histogramPos = threadPos / datasetSize;
        int datasetPos = threadPos % datasetSize;
        distances[threadPos] = 0;

        for (int i = 0; i < 9 * 256; i++) {
            distances[threadPos] += pow((float) (histograms[i + histogramPos * HIST_SIZE]) - dataset[i + datasetPos * HIST_SIZE], 2);
        }
        distances[threadPos] = sqrt(distances[threadPos]);
    }
}

__global__ void CUDA_getKNNDistances(int neighbourCount, float *histogramDistances, float* knnDistances, int datasetSize, int histogramSize) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    if (neighbourCount == 1) {
        knnDistances[threadPos] = histogramDistances[threadPos * datasetSize];
        for (int i = 0; i < datasetSize; i++) {
            if (histogramDistances[(threadPos * datasetSize) + i] < knnDistances[threadPos]) {
                knnDistances[threadPos] = histogramDistances[(threadPos * datasetSize) + i];
            }
        }
    } else {
        float* nearestNeighbours = new float[neighbourCount];
        for (int i = 0; i < neighbourCount; i++) {
            nearestNeighbours[i] = -1;
        }

        for (int i = 0; i < datasetSize; i++) {
            for (int j = 0; j < neighbourCount - 1; j++) {
                if (j == 0) {
                    if ((histogramDistances[(threadPos * datasetSize) + i] < nearestNeighbours[j]) || (nearestNeighbours[j] == -1)) {
                        nearestNeighbours[j] = histogramDistances[(threadPos * datasetSize) + i];
                    }
                }

                if (nearestNeighbours[j] < nearestNeighbours[j+1] || nearestNeighbours[j+1] == -1) {
                    float tmp = nearestNeighbours[j + 1];
                    nearestNeighbours[j+1] = nearestNeighbours[j];
                    nearestNeighbours[j] = tmp;
                } else {
                    break;
                }
            }
        }
        knnDistances[threadPos] = 0;
        for (int i = 0; i < neighbourCount; i++) {
            knnDistances[threadPos] += nearestNeighbours[i];
        }
        delete [] nearestNeighbours;
    }
}

void KNNClassifier::CalculateDistances(int K, float* knnDistances, short* dataset, int datasetSize, short *histograms, int histogramsCount) {
    short* Dev_dataset = nullptr;
    auto err = cudaMalloc((void**)&Dev_dataset, datasetSize * HIST_SIZE * sizeof(short));
    err = cudaMemcpy(Dev_dataset, this->dataset, datasetSize * HIST_SIZE * sizeof(short), cudaMemcpyHostToDevice);
          //cudaMemcpy(Dev_InImg,   imputImg,        width * height,                      cudaMemcpyHostToDevice);

    int biteSize = 500;                        //POCET HISTOGRMU PRO KTERE SE POCITAJI VZDALENOSTI
    int threadCount = biteSize * HIST_SIZE;     //POCET THREADU POTREBNY PRO VYPOCITANI VZDALENOSTI
    int gridSize = (int)ceil(threadCount / 1024);

    float* Dev_distances = nullptr;            //vzdalenosti mezi histogramy a datasetem
    err = cudaMalloc((void**)&Dev_distances, threadCount * datasetSize * sizeof(float));   // ALOKACE PAMETI PRO VZDALENOSTI

    float* Dev_knnDistances = nullptr;         //vzdalenosti ke k nejblizsim sousedum
    err = cudaMalloc((void**)&Dev_knnDistances, threadCount * sizeof(float));   // ALOKACE PAMETI PRO K-NN VZDALENOSTI

    short* Dev_histograms = nullptr;
    err = cudaMalloc((void**)&Dev_histograms, biteSize * HIST_SIZE * sizeof(short));

    int i = 0;
    short* writeFront = histograms;               //pointer pro prenos histogramu na kartu
    float* readFront = knnDistances;           //pointer pro cteni vzdalenosti z karty
    while ((i+1)*biteSize <= histogramsCount) {
        err = cudaMemcpy(Dev_histograms, writeFront, biteSize * HIST_SIZE * sizeof(short), cudaMemcpyHostToDevice);

        dim3 grid_distanceCalculation(gridSize);
        CUDA_calcuateDistances<<<grid_distanceCalculation, 1024>>>(Dev_histograms, Dev_dataset, Dev_distances, datasetSize, biteSize);

        //float* distances = new float[biteSize * datasetSize * HIST_SIZE];

        //err = cudaMemcpy(distances, Dev_distances, biteSize * datasetSize * sizeof(float), cudaMemcpyDeviceToHost);

        dim3 grid_knnDistanceCalc(biteSize);
        CUDA_getKNNDistances<<<grid_knnDistanceCalc, 1>>>(K, Dev_distances, Dev_knnDistances, datasetSize, biteSize);

        err = cudaMemcpy(readFront, Dev_knnDistances, biteSize * sizeof(float), cudaMemcpyDeviceToHost);

        writeFront += HIST_SIZE * biteSize;
        readFront += biteSize;
        i++;
    }

    int restSize = histogramsCount - biteSize * i;

    if (restSize > 0) {
        cudaMemcpy(Dev_histograms, writeFront, restSize * HIST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        int gridSize = (int)ceil(restSize * HIST_SIZE / 1024);
        dim3 grid_distanceCalculation(gridSize);
        CUDA_calcuateDistances<<<grid_distanceCalculation, 1024>>>(Dev_histograms, Dev_dataset, Dev_distances, datasetSize, biteSize);

        dim3 grid_knnDistanceCalc(restSize);
        CUDA_getKNNDistances<<<grid_knnDistanceCalc, 1>>>(K, Dev_distances, Dev_knnDistances, datasetSize, biteSize);

        cudaMemcpy(readFront, Dev_knnDistances, restSize * sizeof(float), cudaMemcpyDeviceToHost);
    }



    cudaFree(Dev_dataset);
    cudaFree(Dev_distances);
    cudaFree(Dev_knnDistances);
    cudaFree(Dev_histograms);
}