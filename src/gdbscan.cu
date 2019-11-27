#include <memory>
#include "dbscan.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

///#include "make_unique.h"


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


__global__ void
adj_list_kernel(size_t* vertex_start_index, size_t* adj_list, float* points_x, float* points_y, float eps, size_t N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) {
        size_t cur_index = vertex_start_index[v];
        float p1_x = points_x[v];
        float p1_y = points_y[v];
        for(size_t i = 0; i < N; i++) {
            float p2_x = points_x[i];
            float p2_y = points_y[i];
            if((p1_x-p2_x)*(p1_x-p2_x) + (p1_y-p2_y)*(p1_y-p2_y) <= eps*eps){
                adj_list[cur_index] = i;
                cur_index++;
            }
        }
    }
}


class ParallelDBScanner: public DBScanner
{
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    size_t scan(
        std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t minPts
    ){ 
        using std::vector;

        std::vector<size_t> vertex_degree(points.size());
        std::vector<size_t> vertex_start_index(points.size());
        std::vector<size_t> adj_list = 
            construct_graph(vertex_degree, vertex_start_index, points, eps);

        setup(vertex_degree.data(), vertex_start_index.data(), adj_list.data(), minPts, vertex_degree.size(), adj_list.size());

        size_t counter = 0;  // current number of clusters
        for(size_t i = 0; i < points.size(); i++){
            int label = labels[i];
            // already in a cluster, skip
            if(label > 0) continue;
            // noise
            if(vertex_degree[i] < minPts) {
                labels[i] = -1;
                continue;
            }
            counter++;
            // BFS
            bfs(i, vertex_degree, vertex_start_index, adj_list, counter, labels, minPts);
        }
        return counter;
    }

private:
    void setup(size_t* vertex_degree, size_t* vertex_start_index, size_t* adj_list, 
                size_t minPts, size_t N, size_t adj_list_len);
    
    bool isEmpty(std::vector<size_t> boarder);

    void bfs(size_t i, 
            std::vector<size_t> &vertex_degree, 
            std::vector<size_t> &vertex_start_index,
            std::vector<size_t> &adj_list, 
            size_t counter, 
            std::vector<int> &labels, 
            size_t minPts);

    std::vector<size_t> construct_graph(std::vector<size_t> &vertex_degree, 
                                        std::vector<size_t> &vertex_start_index, 
                                        std::vector<Vec2> &points, 
                                        float eps);

    void bfs_cuda(size_t* boarder, int* labels, int counter, size_t N);

    void degree_cuda(size_t* vertex_degree, float* points_x, float* points_y, float eps, size_t N);

    void adj_list_cuda(size_t* vertex_start_index, size_t* adj_list, float* points_x, float* points_y, float eps, size_t N, size_t adj_list_len);

    void start_index_cuda(size_t* vertex_start_index, size_t N);

};


// TODO: cudaFree GlobalConstants
void 
ParallelDBScanner::setup(size_t* vertex_degree, size_t* vertex_start_index, size_t* adj_list, size_t minPts, size_t N, size_t adj_list_len) {
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


bool 
ParallelDBScanner::isEmpty(std::vector<size_t> boarder) {
    for(size_t i = 0; i < boarder.size(); i++) {
        if(boarder[i]) {
            return false;
        }
    }
    return true;
}


void 
ParallelDBScanner::bfs(size_t i, 
        std::vector<size_t> &vertex_degree, 
        std::vector<size_t> &vertex_start_index,
        std::vector<size_t> &adj_list, 
        size_t counter, 
        std::vector<int> &labels, 
        size_t minPts) {
    // TODO: use bit vector
    std::vector<size_t> boarder(vertex_degree.size(), 0);
    boarder[i] = 1;
    while(!isEmpty(boarder)) {
        bfs_cuda(boarder.data(), labels.data(), counter, vertex_degree.size());
    }
}



std::vector<size_t> 
ParallelDBScanner::construct_graph(std::vector<size_t> &vertex_degree, 
                                    std::vector<size_t> &vertex_start_index, 
                                    std::vector<Vec2> &points, 
                                    float eps) {
    std::vector<float> points_x(points.size());
    std::vector<float> points_y(points.size());
    for(size_t i = 0; i < points.size(); i++) {
        points_x[i] = points[i].x;
        points_y[i] = points[i].y;
    }
    degree_cuda(vertex_degree.data(), points_x.data(), points_y.data(), eps, points.size());

    vertex_start_index = vertex_degree;
    start_index_cuda(vertex_start_index.data(), points.size());
    size_t cur_index = vertex_start_index[points.size()-1] + vertex_degree[points.size()-1];

    std::vector<size_t> adj_list(cur_index);
    adj_list_cuda(vertex_start_index.data(), adj_list.data(), points_x.data(), points_y.data(), eps, points.size(), cur_index);
    return adj_list;
}


void 
ParallelDBScanner::bfs_cuda(size_t* boarder, int* labels, int counter, size_t N) {
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


void 
ParallelDBScanner::degree_cuda(size_t* vertex_degree, float* points_x, float* points_y, float eps, size_t N) {
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


void 
ParallelDBScanner::adj_list_cuda(size_t* vertex_start_index, size_t* adj_list, float* points_x, float* points_y, float eps, size_t N, size_t adj_list_len) {
    int bytes_start_index = sizeof(size_t) * N;
    int bytes_adj_list = sizeof(size_t) * adj_list_len;
    int bytes_points = sizeof(float) * N;

    // compute number of blocks and threads per block
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    size_t* device_start_index;
    size_t* device_adj_list;
    float* device_points_x;
    float* device_points_y;

    cudaMalloc(&device_start_index, bytes_start_index);
    cudaMalloc(&device_adj_list, bytes_adj_list);
    cudaMalloc(&device_points_x, bytes_points);
    cudaMalloc(&device_points_y, bytes_points);

    cudaMemcpy(device_start_index, vertex_start_index, bytes_start_index, cudaMemcpyHostToDevice);
    cudaMemcpy(device_points_x, points_x, bytes_points, cudaMemcpyHostToDevice);
    cudaMemcpy(device_points_y, points_y, bytes_points, cudaMemcpyHostToDevice);

    adj_list_kernel<<<blocks, threadsPerBlock>>>(device_start_index, device_adj_list, device_points_x, device_points_y, eps, N);

    cudaThreadSynchronize();
    cudaMemcpy(adj_list, device_adj_list, bytes_adj_list, cudaMemcpyDeviceToHost);

    cudaFree(device_start_index);
    cudaFree(device_adj_list);
    cudaFree(device_points_x);
    cudaFree(device_points_y);
}


void 
ParallelDBScanner::start_index_cuda(size_t* vertex_start_index, size_t N) {
    int bytes = sizeof(size_t) * N;
    size_t* device_start_index;
    
    cudaMalloc(&device_start_index, bytes);
    cudaMemcpy(device_start_index, vertex_start_index, bytes, cudaMemcpyHostToDevice);

    thrust::device_ptr<size_t> device_start_index_ptr = thrust::device_pointer_cast(device_start_index);
    thrust::exclusive_scan(device_start_index_ptr, device_start_index_ptr + N, device_start_index_ptr);

    cudaMemcpy(vertex_start_index, device_start_index, bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_start_index);
}

std::unique_ptr<DBScanner> createParallelDBScanner(){
    return std::make_unique<ParallelDBScanner>();
}


