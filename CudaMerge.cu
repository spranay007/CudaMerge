#include <iostream>
#include <helper_cuda.h>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_functions.h>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <random>

#define THREADS_PER_BLOCK 256

/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its running time
 * is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 * running on the GPU may provide a significant boost in performance
 */

 // helper for main()
long readList(long**);

// data[], size, threads
void mergesort(long*, long, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);
void cpu_mergesort(long* data, long size);
void merge(long* result, long* left, long* right, long size_left, long size_right);

// profiling
long long tm();

#define min(a, b) (a < b ? a : b)

// Generate random numbers
void generateRandomNumbers(std::vector<long>& numbers, long size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<long> dis(1, 1000);

    for (long i = 0; i < size; ++i) {
        numbers.push_back(dis(gen));
    }
}

void printHelp(char* program) {
    std::cout << "usage: " << program << " <number_of_elements>\n";
}

int main(int argc, char** argv) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK, 1, 1);

    // Check if the number of elements is provided
    if (argc < 2) {
        std::cout << "Number of elements not provided.\n";
        printHelp(argv[0]);
        return -1;
    }

    long size = std::atol(argv[1]);
    if (size <= 0) {
        std::cout << "Invalid number of elements.\n";
        return -1;
    }

    // Generate random numbers
    std::vector<long> numbers;
    generateRandomNumbers(numbers, size);

    dim3 blocksPerGrid((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1, 1);

    // GPU Sorting
    auto start_gpu = std::chrono::steady_clock::now();
    // Perform GPU mergesort
    mergesort(numbers.data(), size, threadsPerBlock);
    auto end_gpu = std::chrono::steady_clock::now();
    auto gpu_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu).count();

    // CPU Sorting
    auto start_cpu = std::chrono::steady_clock::now();
    // Perform CPU mergesort
    cpu_mergesort(numbers.data(), size);
    auto end_cpu = std::chrono::steady_clock::now();
    auto cpu_sort_time = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count();

    std::cout << "GPU Sorting Time: " << gpu_sort_time << " microseconds\n";
    std::cout << "CPU Sorting Time: " << cpu_sort_time << " microseconds\n";

    return 0;
}

void mergesort(long* data, long size, dim3 threadsPerBlock) {
    long* D_data;
    long* D_swp;

    checkCudaErrors(cudaMalloc((void**)&D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**)&D_swp, size * sizeof(long)));

    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));

    gpu_mergesort << <1, threadsPerBlock >> > (D_data, D_swp, size, 2, 1);

    checkCudaErrors(cudaMemcpy(data, D_data, size * sizeof(long), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(D_data));
    checkCudaErrors(cudaFree(D_swp));
}

__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    long start = width * idx * slices,
        middle,
        end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        }
        else {
            dest[k] = source[j];
            j++;
        }
    }
}

long long tm() {
    static std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastTime);
    long long t = elapsedMicroseconds.count();
    lastTime = currentTime;
    return t;
}

void cpu_mergesort(long* data, long size) {
    if (size <= 1)
        return;

    long mid = size / 2;

    cpu_mergesort(data, mid);
    cpu_mergesort(data + mid, size - mid);

    merge(data, data, data + mid, mid, size - mid);
}

void merge(long* result, long* left, long* right, long size_left, long size_right) {
    long i = 0, j = 0, k = 0;
    std::vector<long> merged(size_left + size_right);

    while (i < size_left && j < size_right) {
        if (left[i] <= right[j])
            merged[k++] = left[i++];
        else
            merged[k++] = right[j++];
    }

    while (i < size_left)
        merged[k++] = left[i++];

    while (j < size_right)
        merged[k++] = right[j++];

    std::copy(merged.begin(), merged.end(), result);
}
