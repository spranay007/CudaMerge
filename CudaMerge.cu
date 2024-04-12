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

#define BLOCK_SIZE 256

// mergesort.cu
// a one-file c++ / cuda program for performing mergesort on the GPU
// While the program execution is fairly slow, most of its runnning time
// is spent allocating memory on the GPU.
// For a more complex program that performs many calculations,
// running on the GPU may provide a significant boost in performance

// helper for main()
long readList(long**);

// data[], size, threads, blocks,
void mergesort(long*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
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

    std::cout
        << "usage: " << program << " [-xyzXYZv]\n"
        << '\n'
        << "-x, -y, -z are numbers of threads in each dimension. On my machine\n"
        << "  the correct number is x*y*z = 32\n"
        << '\n'
        << "-X, -Y, -Z are numbers of blocks to use in each dimension. Each block\n"
        << "  holds x*y*z threads, so the total number of threads is:\n"
        << "  x*y*z*X*Y*Z\n"
        << '\n'
        << "-v prints out extra info\n"
        << '\n'
        << "? prints this message and exits\n"
        << '\n'
        << "example: ./mergesort -x 8 -Y 10 -v\n"
        << '\n'
        << "Reads in a list of integer numbers from stdin, and performs\n"
        << "a bottom-up merge sort:\n"
        << '\n'
        << "Input:          8 3 1 9 1 2 7 5 9 3 6 4 2 0 2 5\n"
        << "Threads: |    t1    |    t2    |    t3    |    t4    |\n"
        << "         | 8 3 1 9  | 1 2 7 5  | 9 3 6 4  | 2 0 2 5  |\n"
        << "         |  38 19   |  12 57   |  39 46   |  02 25   |\n"
        << "         |   1398   |   1257   |   3469   |   0225   |\n"
        << "         +----------+----------+----------+----------+\n"
        << "         |          t1         |          t2         |\n"
        << "         |       11235789      |       02234569      |\n"
        << "         +---------------------+---------------------+\n"
        << "         |                     t1                    |\n"
        << "         |      0 1 1 2 2 2 3 3 4 5 5 6 7 8 9 9      |\n"
        << '\n'
        << '\n';
}

bool verbose;
int main(int argc, char** argv) {
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = BLOCK_SIZE;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    bool verbose = false;

    // Parse argv
    /*
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = nullptr;
            switch (arg) {
            case 'x':
                toSet = &threadsPerBlock.x;
                break;
            case 'y':
                toSet = &threadsPerBlock.y;
                break;
            case 'z':
                toSet = &threadsPerBlock.z;
                break;
            case 'X':
                toSet = &blocksPerGrid.x;
                break;
            case 'Y':
                toSet = &blocksPerGrid.y;
                break;
            case 'Z':
                toSet = &blocksPerGrid.z;
                break;
            case 'v':
                verbose = true;
                break;
            default:
                std::cout << "unknown argument: " << arg << '\n';
                printHelp(argv[0]);
                return -1;
            }

            if (toSet) {
                i++;
                *toSet = static_cast<unsigned int>(strtol(argv[i], nullptr, 10));
            }
        }
        else {
            if (argv[i][0] == '?' && !argv[i][1])
                std::cout << "help:\n";
            else
                std::cout << "invalid argument: " << argv[i] << '\n';
            printHelp(argv[0]);
            return -1;
        }
    }

    if (verbose) {
        std::cout << "\nthreadsPerBlock:"
            << "\n  x: " << threadsPerBlock.x
            << "\n  y: " << threadsPerBlock.y
            << "\n  z: " << threadsPerBlock.z
            << "\n\nblocksPerGrid:"
            << "\n  x:" << blocksPerGrid.x
            << "\n  y:" << blocksPerGrid.y
            << "\n  z:" << blocksPerGrid.z
            << "\n\n total threads: "
            << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
            blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z
            << "\n\n";
    }
    */
    // Check if the number of numbers is provided
    if (argc < 2) {
        std::cout << "Number of numbers not provided.\n";
        return -1;
    }

    long size = std::atol(argv[1]);
    if (size <= 0) {
        std::cout << "Invalid number of numbers.\n";
        return -1;
    }

    // Generate random numbers
    std::vector<long> numbers;
    generateRandomNumbers(numbers, size);

    // GPU Sorting
    auto start_gpu = std::chrono::steady_clock::now();
    // Perform GPU mergesort
    mergesort(numbers.data(), size, threadsPerBlock, blocksPerGrid);
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

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;

    // Actually allocate the two arrays
    tm();
    checkCudaErrors(cudaMalloc((void**)&D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**)&D_swp, size * sizeof(long)));
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
    if (verbose)
        std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";

    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void**)&D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
        blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads)*width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width
                << ", slices: " << slices
                << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort << <blocksPerGrid, threadsPerBlock >> > (A, B, size, width, slices, D_threads, D_blocks);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    tm();
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";


    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
        threadIdx.y * (x = threads->x) +
        threadIdx.z * (x *= threads->y) +
        blockIdx.x * (x *= threads->z) +
        blockIdx.y * (x *= blocks->z) +
        blockIdx.z * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
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

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    __shared__ long shared[BLOCK_SIZE];
    long i = start + threadIdx.x;
    long j = middle + threadIdx.x;
    long index = start + threadIdx.x;

    // Load the data to shared memory
    if (i < middle) shared[threadIdx.x] = source[i];
    else shared[threadIdx.x] = 0;

    __syncthreads();

    // Merge
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || shared[threadIdx.x] < source[j])) {
            dest[index] = shared[threadIdx.x];
            i++;
        }
        else {
            dest[index] = source[j];
            j++;
        }
        index += blockDim.x;
    }
}

// read data into a minimal linked list
typedef struct {
    int v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(long** list) {
    tm();
    long v, size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (std::cin >> v) {
        LinkNode* next = new LinkNode();
        next->v = v;
        if (node)
            node->next = next;
        else
            first = next;
        node = next;
        size++;
    }


    if (size) {
        *list = new long[size];
        LinkNode* node = first;
        long i = 0;
        while (node) {
            (*list)[i++] = node->v;
            node = (LinkNode*)node->next;
        }

    }

    if (verbose)
        std::cout << "read stdin: " << tm() << " microseconds\n";

    return size;
}


// 
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
//struct timeval tStart;
//int tm() {
    //  struct timeval tEnd;
    //  gettimeofday(&tEnd, 0);
    //  int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    //  tStart = tEnd;
//    return 100;
//}
// 
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
std::chrono::steady_clock::time_point lastTime = std::chrono::steady_clock::now();

long long tm() {
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedMicroseconds = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastTime);
    long long t = elapsedMicroseconds.count();
    lastTime = currentTime;
    return t;
}

// CPU-based merge sort
void cpu_mergesort(long* data, long size) {
    if (size <= 1)
        return;

    long mid = size / 2;

    cpu_mergesort(data, mid);
    cpu_mergesort(data + mid, size - mid);

    // Merge the two sorted halves
    merge(data, data, data + mid, mid, size - mid);
}

// Merge function for CPU merge sort
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

    // Copy the merged array back to the result
    std::copy(merged.begin(), merged.end(), result);
}
