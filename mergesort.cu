#include <thrust/generate.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cmath>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>

using namespace std;

class RandomNumber {
	int nLimit;
public:
	RandomNumber(int lim) : nLimit(lim) {}
	float operator() () { return (rand() % nLimit); }
};


//  CPU

void merge(int* list, int* sorted, int start, int mid, int end)
{
	int ti = start, i = start, j = mid;
	while (i < mid || j < end)
	{
		if (j == end) sorted[ti] = list[i++];
		else if (i == mid) sorted[ti] = list[j++];
		else if (list[i] < list[j]) sorted[ti] = list[i++];
		else sorted[ti] = list[j++];
		ti++;
	}

	for (ti = start; ti < end; ti++)
		list[ti] = sorted[ti];
}

void mergesort_recur(int* list, int* sorted, int start, int end)
{
	if (end - start < 2)
		return;

	mergesort_recur(list, sorted, start, start + (end - start) / 2);
	mergesort_recur(list, sorted, start + (end - start) / 2, end);
	merge(list, sorted, start, start + (end - start) / 2, end);
}

int mergesort_cpu(int* list, int* sorted, int n)
{
	mergesort_recur(list, sorted, 0, n);
	return 1;
}


//  GPU Implementation 

__device__ void merge_gpu(int* list, int* sorted, int start, int mid, int end)
{
	int k = start, i = start, j = mid;
	while (i < mid || j < end)
	{
		if (j == end)
		{
			sorted[k] = list[i++];
		}
		else if (i == mid) sorted[k] = list[j++];
		else if (list[i] < list[j]) sorted[k] = list[i++];
		else sorted[k] = list[j++];
		k++;
	}
}

__global__ void mergesort_gpu(int* list, int* sorted, int n, int chunk)
{

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int start = tid * chunk;
	if (start >= n) return;
	int mid, end;

	mid = min(start + chunk / 2, n);
	end = min(start + chunk, n);
	merge_gpu(list, sorted, start, mid, end);
}

// Sequential Merge Sort for GPU when Number of Threads Required gets below 1 Warp Size
void mergesort_gpu_seq(int* list, int* sorted, int n, int chunk)
{
	int chunk_id;
	for (chunk_id = 0; chunk_id * chunk <= n; chunk_id++) {
		int start = chunk_id * chunk, end, mid;
		if (start >= n) return;
		mid = min(start + chunk / 2, n);
		end = min(start + chunk, n);
		merge(list, sorted, start, mid, end);
	}
}


int mergesort(int* list, int* sorted, int n)
{

	int* list_d;
	int* sorted_d;
	int dummy;
	bool flag = false;
	bool sequential = false;

	int size = n * sizeof(int);

	cudaMalloc((void**)&list_d, size);
	cudaMalloc((void**)&sorted_d, size);

	cudaMemcpy(list_d, list, size, cudaMemcpyHostToDevice);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error_2: %s\n", cudaGetErrorString(err));
		return -1;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	// compute capability
	const int max_active_blocks_per_sm = 16;
	const int max_active_warps_per_sm = 64;

	int warp_size = prop.warpSize;
	int max_grid_size = prop.maxGridSize[0];
	int max_threads_per_block = prop.maxThreadsPerBlock;
    std::cout << "max_threads_per_block" << max_threads_per_block << std::endl;
	int max_procs_count = prop.multiProcessorCount;

	int max_active_blocks = max_active_blocks_per_sm * max_procs_count;
	int max_active_warps = max_active_warps_per_sm * max_procs_count;

	int chunk_size;
	for (chunk_size = 2; chunk_size < 2 * n; chunk_size *= 2)
	{
		int blocks_required = 0, threads_per_block = 0;
		int threads_required = (n % chunk_size == 0) ? n / chunk_size : n / chunk_size + 1;
        
		if (threads_required <= warp_size * 3 && !sequential)
		{
//             std::cout << "COPY DATA TO HOST   ";
//             std::cout << chunk_size << "- if (threads_required[" << threads_required << "]<=warp_size*3[" << warp_size * 3 << "]&& !sequential[" << !sequential << "])" << std::endl;
			sequential = true;
			if (flag)
			{
				cudaMemcpy(list, sorted_d, size, cudaMemcpyDeviceToHost);
			}
			else
			{
				cudaMemcpy(list, list_d, size, cudaMemcpyDeviceToHost);
			}
			err = cudaGetLastError();
			if (err != cudaSuccess)
			{
				printf("ERROR_4: %s\n", cudaGetErrorString(err));
				return -1;
			}
			cudaFree(list_d);
			cudaFree(sorted_d);
		}
		else if (threads_required < max_threads_per_block)
		{
			threads_per_block = warp_size * 4;
			dummy = threads_required / threads_per_block;
			blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
		}
		else if (threads_required < max_active_blocks * warp_size * 4)
		{
			threads_per_block = max_threads_per_block / 2;
			dummy = threads_required / threads_per_block;
			blocks_required = (threads_required % threads_per_block == 0) ? dummy : dummy + 1;
		}
		else
		{
			dummy = threads_required / max_active_blocks;
			int estimated_threads_per_block = (threads_required % max_active_blocks == 0) ? dummy : dummy + 1;
			if (estimated_threads_per_block > max_threads_per_block)
			{
				threads_per_block = max_threads_per_block;
				dummy = threads_required / max_threads_per_block;
				blocks_required = (threads_required % max_threads_per_block == 0) ? dummy : dummy + 1;
			}
			else
			{
				threads_per_block = estimated_threads_per_block;
				blocks_required = max_active_blocks;
			}
		}

		if (blocks_required >= max_grid_size)
		{
			printf("ERROR_2: Too many Blocks Required\n");
			return -1;
		}

		if (sequential)
		{
//             std::cout << "cpu merge" << std::endl;
			mergesort_gpu_seq(list, sorted, n, chunk_size);
		}
		else
		{
//             std::cout << "GPU RUN" << std::endl;
			if (flag) mergesort_gpu << <blocks_required, threads_per_block >> > (sorted_d, list_d, n, chunk_size);
			else mergesort_gpu << <blocks_required, threads_per_block >> > (list_d, sorted_d, n, chunk_size);
			cudaDeviceSynchronize();
			err = cudaGetLastError();
			if (err != cudaSuccess) {
				printf("ERROR_3: %s\n", cudaGetErrorString(err));
				return -1;
			}
			flag = !flag;
		}
//         std::cout << "chunk_size: " << chunk_size << " blocks_required: " << blocks_required << " threads_per_block: " << threads_per_block << " threads_required: " << threads_required << "    " << flag << std::endl;
	}
	return 0;
}

int main(int argc, char const *argv[]) {

	// struct timespec start, stop;
	clock_t start, stop, result;
// 	int n_list[] = { 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 100000000};
	int n_list[] = { 100000000, 1000000000, 100000000 };
    int i, j;
	for (j = 0; j < 3; j++) {
		printf("Elements count: %d\n", n_list[j]);
		int *sorted = (int *)malloc(n_list[j] * sizeof(int));
		int *list = (int *)malloc(n_list[j] * sizeof(int));
		int *sorted_s = (int *)malloc(n_list[j] * sizeof(int));
		int *list_s = (int *)malloc(n_list[j] * sizeof(int));
		for (i = 0; i < n_list[j]; i++) {
			list[i] = rand() % 10000;
			list_s[i] = list[i];
		}
		start = clock();

		mergesort(list, sorted, n_list[j]);

		stop = clock();
		cout << "GPU time: " << (long double)(stop - start) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;

		start = clock();

		mergesort_cpu(list_s, sorted_s, n_list[j]);

		stop = clock();

		cout << "CPU time: " << (long double)(stop - start) / (CLOCKS_PER_SEC / 1000) << " ms" << endl;

		for (i = 1; i < n_list[j]; i++) {
			if (sorted[i - 1] > sorted[i]) {
				printf("false sort _1\n");
				return -1;
			}
		}
		for (i = 0; i < n_list[j]; i++) {
			if (sorted_s[i] != sorted[i]) {
				printf("false sort _2\n");
				printf("P:%d, S:%d, Index:%d\n", sorted[i], sorted_s[i], i);
				return -1;
			}
		}
		printf("corrected sort\n");

		free(sorted);
		free(list);
		free(sorted_s);
		free(list_s);
		printf("---------------------------------------------------\n");
	}
	return 0;
}
