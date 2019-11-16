#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void
bfs_kernel(size_t* vertex_degree,
			size_t* vertex_start_index,
			size_t* adj_list,
			size_t* boarder,
			size_t minPts,
			int* labels,
			int counter,
			size_t N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
    	if(boarder[j]) {
            boarder[j] = 0;
            labels[j] = counter;
            if(vertex_degree[j] < minPts) {
                return;
            }
            size_t start_index = vertex_start_index[j];
            size_t end_index = start_index + vertex_degree[j];
            for(size_t neighbor_index = start_index; 
                neighbor_index < end_index; 
                neighbor_index++) {
                size_t neighbor = adj_list[neighbor_index];
                if(labels[neighbor] <= 0) {
                    boarder[neighbor] = 1;
                }
            }
        }
    }
}

void bfs_cuda(size_t* vertex_degree, 
                size_t* vertex_start_index,
                size_t* adj_list, 
                size_t* boarder,  
                size_t minPts, 
                int* labels, 
                int counter,
                size_t N,
                size_t adj_list_len) {

	int bytes = sizeof(size_t) * N;
	int adj_list_bytes = sizeof(size_t) * adj_list_len;
	int labels_byte = sizeof(int) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t* device_degree;
    size_t* device_start_index;
    size_t* device_adj_list;
    size_t* device_boarder;
    int* device_labels;

    cudaMalloc(&device_degree, bytes);
    cudaMalloc(&device_start_index, bytes);
    cudaMalloc(&device_adj_list, adj_list_bytes);
    cudaMalloc(&device_boarder, bytes);
    cudaMalloc(&device_labels, labels_byte);

    cudaMemcpy(device_degree, vertex_degree, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_start_index, vertex_start_index, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_adj_list, adj_list, adj_list_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_boarder, boarder, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_labels, labels, labels_byte, cudaMemcpyHostToDevice);

    bfs_kernel<<<blocks, threadsPerBlock>>>(device_degree, device_start_index, device_adj_list, device_boarder, minPts, device_labels, counter, N);

    cudaThreadSynchronize();
    cudaMemcpy(boarder, device_boarder, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, device_labels, labels_byte, cudaMemcpyDeviceToHost);

    cudaFree(device_degree);
    cudaFree(device_start_index);
    cudaFree(device_adj_list);
    cudaFree(device_boarder);
    cudaFree(device_labels);
}

