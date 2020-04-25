//
// Created by pavel on 25.04.20.
//

#include "KNNClassifier.h"
#include "definitions.h"

__global__ void CUDA_calcuateDistances(int* histograms, int*dataset, double* distances, int datasetSize, int histogramSize) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadPos < histogramSize * datasetSize) {
        int histogramPos = threadPos / datasetSize;
        int datasetPos = threadPos % datasetSize;
        distances[threadPos] = 0;

        for (int i = 0; i < 9 * 256; i++) {
            distances[threadPos] += pow(
                    (double) (histograms[i + histogramPos * HIST_SIZE]) - dataset[i + datasetPos * HIST_SIZE], 2);
        }
        distances[threadPos] = sqrt(distances[threadPos]);
    }
}

__global__ void CUDA_getKNNDistances(int neighbourCount, double*histogramDistances, double*knnDistances, int datasetSize, int histogramSize) {
    long threadPos = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadPos < datasetSize * histogramSize) {
        if (neighbourCount == 1) {          //1-NN
            knnDistances[threadPos] = -1;
            for (int i = 0; i < datasetSize; i++) {
                double dst = histogramDistances[i + threadPos * datasetSize];
                if (knnDistances[threadPos] == -1 || knnDistances[threadPos] > dst) {
                    knnDistances[threadPos] = dst;
                }
            }
        } else {                            //N-NN
            double *nearestNeighbours = new double[neighbourCount];
            for (int i = 0; i < neighbourCount; i++) {
                nearestNeighbours[i] = -1;
            }

            for (int i = 0; i < datasetSize; i++) {

                for (int j = 0; j < neighbourCount - 1; j++) { // nespusti se pro k= 1
                    if (j == 0) {
                        double dst = histogramDistances[i + threadPos * datasetSize];
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
            for (int i = 0; i < neighbourCount; i++) {
                knnDistances[threadPos] += nearestNeighbours[i];
            }
        }
    }
}

void KNNClassifier::CalculateDistances(int K, double *knnDistances, int *dataset, int datasetSize, int *histograms, int histogramsCount) {
    int* Dev_dataset = nullptr;
    cudaMalloc((void**)&Dev_dataset, datasetSize * HIST_SIZE * sizeof(int));
    cudaMemcpy(Dev_dataset, dataset, datasetSize * HIST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    int biteSize = 1000;                        //POCET HISTOGRMU PRO KTERE SE POCITAJI VZDALENOSTI
    int threadCount = biteSize * HIST_SIZE;     //POCET THREADU POTREBNY PRO VYPOCITANI VZDALENOSTI
    int gridSize = (int)ceil(threadCount / 1024);

    double* Dev_distances = nullptr;            //vzdalenosti mezi histogramy a datasetem
    cudaMalloc((void**)&Dev_distances, threadCount * datasetSize * sizeof(double));   // ALOKACE PAMETI PRO VZDALENOSTI

    double* Dev_knnDistances = nullptr;         //vzdalenosti ke k nejblizsim sousedum
    cudaMalloc((void**)&Dev_knnDistances, threadCount * sizeof(double));   // ALOKACE PAMETI PRO K-NN VZDALENOSTI

    int* Dev_histograms = nullptr;
    cudaMalloc((void**)&Dev_histograms, biteSize * HIST_SIZE * sizeof(int));

    int i = 0;
    int* writeFront = histograms;               //pointer pro prenos histogramu na kartu
    double* readFront = knnDistances;           //pointer pro cteni vzdalenosti z karty
    while ((i+1)*biteSize <= histogramsCount) {
        cudaMemcpy(Dev_histograms, writeFront, biteSize * HIST_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        dim3 grid_distanceCalculation(gridSize);
        CUDA_calcuateDistances<<<grid_distanceCalculation, 1024>>>(Dev_histograms, Dev_dataset, Dev_distances, datasetSize, biteSize);

        dim3 grid_knnDistanceCalc(biteSize);
        CUDA_getKNNDistances<<<grid_knnDistanceCalc, 1>>>(K, Dev_distances, Dev_knnDistances, datasetSize, biteSize);

        cudaMemcpy(knnDistances, Dev_distances, biteSize * biteSize * sizeof(double), cudaMemcpyDeviceToHost);

        writeFront += HIST_SIZE * biteSize;
        readFront += biteSize;
        i++;
    }
    //TODO -> calculate rest

    cudaFree(Dev_dataset);
    cudaFree(Dev_distances);
    cudaFree(Dev_knnDistances);
    cudaFree(Dev_histograms);
}