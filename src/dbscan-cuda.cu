#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>


struct GlobalConstants {
    size_t* vertex_degree;
    size_t* vertex_start_index;
    size_t* adj_list;
    size_t minPts;
    size_t N;
};

__constant__ GlobalConstants cuConstParams;

__global__ void
bfs_kernel(size_t* boarder, int* labels, int counter) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cuConstParams.N) {
    	if(boarder[j]) {
            boarder[j] = 0;
            labels[j] = counter;
            if(cuConstParams.vertex_degree[j] < cuConstParams.minPts) {
                return;
            }
            size_t start_index = cuConstParams.vertex_start_index[j];
            size_t end_index = start_index + cuConstParams.vertex_degree[j];
            for(size_t neighbor_index = start_index; 
                neighbor_index < end_index; 
                neighbor_index++) {
                size_t neighbor = cuConstParams.adj_list[neighbor_index];
                if(labels[neighbor] <= 0) {
                    boarder[neighbor] = 1;
                }
            }
        }
    }
}

__global__ void
degree_kernel(size_t* vertex_degree, float* points_x, float* points_y, float eps, size_t N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) {
        size_t degree = 0;
        float p1_x = points_x[v];
        float p1_y = points_y[v];
        for(size_t i = 0; i < N; i++){
            float p2_x = points_x[i];
            float p2_y = points_y[i];
            if((p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y) <= eps*eps){
                degree++;
            }
        }
        vertex_degree[v] = degree;
    }
}

void setup(size_t* vertex_degree, size_t* vertex_start_index, size_t* adj_list, size_t minPts, size_t N, size_t adj_list_len) {
    int bytes = sizeof(size_t) * N;
    int adj_list_bytes = sizeof(size_t) * adj_list_len;

    size_t* device_degree;
    size_t* device_start_index;
    size_t* device_adj_list;

    cudaMalloc(&device_degree, bytes);
    cudaMalloc(&device_start_index, bytes);
    cudaMalloc(&device_adj_list, adj_list_bytes);

    cudaMemcpy(device_degree, vertex_degree, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_start_index, vertex_start_index, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_adj_list, adj_list, adj_list_bytes, cudaMemcpyHostToDevice);

    GlobalConstants params;
    params.vertex_degree = device_degree;
    params.vertex_start_index = device_start_index;
    params.adj_list = device_adj_list;
    params.minPts = minPts;
    params.N = N;

    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}

// TODO: cudaFree GlobalConstants

void bfs_cuda(size_t* boarder, int* labels, int counter, size_t N) {
    int bytes = sizeof(size_t) * N;
	int labels_byte = sizeof(int) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t* device_boarder;
    int* device_labels;

    cudaMalloc(&device_boarder, bytes);
    cudaMalloc(&device_labels, labels_byte);

    cudaMemcpy(device_boarder, boarder, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_labels, labels, labels_byte, cudaMemcpyHostToDevice);

    bfs_kernel<<<blocks, threadsPerBlock>>>(device_boarder, device_labels, counter);

    cudaThreadSynchronize();
    cudaMemcpy(boarder, device_boarder, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, device_labels, labels_byte, cudaMemcpyDeviceToHost);

    cudaFree(device_boarder);
    cudaFree(device_labels);
}

void degree_cuda(size_t* vertex_degree, float* points_x, float* points_y, float eps, size_t N) {
    int bytes_degree = sizeof(size_t) * N;
    int bytes_points = sizeof(float) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t* device_degree;
    float* device_points_x;
    float* device_points_y;

    cudaMalloc(&device_degree, bytes_degree);
    cudaMalloc(&device_points_x, bytes_points);
    cudaMalloc(&device_points_y, bytes_points);

    cudaMemcpy(device_points_x, points_x, bytes_points, cudaMemcpyHostToDevice);
    cudaMemcpy(device_points_y, points_y, bytes_points, cudaMemcpyHostToDevice);

    degree_kernel<<<blocks, threadsPerBlock>>>(device_degree, device_points_x, device_points_y, eps, N);

    cudaThreadSynchronize();
    cudaMemcpy(vertex_degree, device_degree, bytes_degree, cudaMemcpyDeviceToHost);

    cudaFree(device_degree);
    cudaFree(device_points_x);
    cudaFree(device_points_y);

}

