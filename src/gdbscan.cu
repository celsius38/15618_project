#include <memory>
#include "dbscan.h"
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "make_unique.h"

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
#if defined(DEBUG)
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
#endif
} 

struct GlobalConstants {
    float squared_eps;
    size_t min_points;
    size_t num_points;
    size_t adj_list_size;

    float* points;
    size_t* degree;
    size_t* start_index;
    size_t* adj_list;
};

__constant__ GlobalConstants cuConstParams;

const int THREADS_PER_BLOCK = 512;


__global__ void
check_cuda_const(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0){
        printf("cuda: sizeof(float) = %ld\n", (unsigned long)sizeof(float));
        printf("cuda: sizeof(size_t) = %ld\n", (unsigned long)sizeof(size_t));
        printf("cuda: squared_eps: %f, min_points: %lu, " 
                "num_points: %lu, adj_list_size: %lu\n" 
                "points:%p, degree: %p, start_index: %p, adj_list: %p\n",
                (float)cuConstParams.squared_eps, 
                (size_t)cuConstParams.min_points, 
                (size_t)cuConstParams.num_points, 
                (size_t)cuConstParams.adj_list_size,
                (void*)cuConstParams.points,
                (void*)cuConstParams.degree, 
                (void*)cuConstParams.start_index,
                (void*)cuConstParams.adj_list); 
        printf("cuda: points[0]: (%f,%f), points[999]: (%f, %f)\n", 
                cuConstParams.points[0], 
                cuConstParams.points[1], 
                cuConstParams.points[1998], 
                cuConstParams.points[1999]);
        printf("cuda: degree[0]: %d, degree[999]: %d\n", 
                cuConstParams.degree[0],
                cuConstParams.degree[999]);
    }
}


__device__ float 
squared_distance(float2 p1, float2 p2){
    float x = p1.x - p2.x;
    float y = p1.y - p2.y;
    return sqrt(x*x + y*y);
}

class ParallelDBScanner: public DBScanner
{
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    ParallelDBScanner();
    ~ParallelDBScanner();
    size_t scan(
        std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t min_points
    );

private:
    float squared_eps;
    size_t min_points;
    size_t num_points;
    size_t adj_list_size;

    size_t* host_degree;

    float* device_points;
    size_t* device_degree;
    size_t* device_start_index;
    size_t* device_adj_list;

    void setup(std::vector<Vec2> &points, float eps, size_t min_points);
    
    bool isEmpty(std::vector<size_t>& border);

    void bfs(size_t i, size_t counter, std::vector<int>& labels);

    void construct_graph();

    void bfs_cuda(size_t* border, int* labels, int counter);

    void check_scanner_const();
};

ParallelDBScanner::ParallelDBScanner(){
    num_points = 0;
    adj_list_size = 0;
    host_degree = NULL;

    device_points = NULL;
    device_degree = NULL;
    device_start_index = NULL;
    device_adj_list = NULL;
    squared_eps = 0.f;
    min_points = 0;
}


ParallelDBScanner::~ParallelDBScanner(){
    if(host_degree) delete[] host_degree;

    if(device_points) cudaFree(device_points);
    if(device_degree) cudaFree(device_degree);
    if(device_start_index) cudaFree(device_start_index);
    if(device_adj_list) cudaFree(device_adj_list);
}


void
ParallelDBScanner::setup(std::vector<Vec2> &points, float eps, size_t min_points)
{
    // allocate host data
    this -> squared_eps = eps * eps;
    this -> min_points = min_points;
    this -> num_points = points.size();
    this -> host_degree = new size_t[points.size()];

    // allocate device data
    size_t points_bytes = sizeof(float) * 2 * num_points;
    size_t bytes = sizeof(size_t) * num_points;
    cudaMalloc(&device_points, points_bytes);
    cudaMalloc(&device_degree, bytes);
    cudaMalloc(&device_start_index, bytes);
   
    // copy data and hyper parameters to device 
    cudaMemcpy(device_points, (float*)points.data(), points_bytes, cudaMemcpyHostToDevice); 
    GlobalConstants params;
    params.squared_eps = squared_eps;
    params.min_points = min_points;
    params.num_points = num_points;
    params.adj_list_size = 0;
    params.points = device_points;
    params.degree = device_degree;
    params.start_index = device_start_index; 
    params.adj_list = NULL;
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}


size_t 
ParallelDBScanner::scan(
    std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t min_points
)
{
    setup(points, eps, min_points);
    construct_graph();

    size_t counter = 0;  // current number of clusters
    for(size_t i = 0; i < points.size(); i++){
        // already in a cluster, skip
        if(labels[i] > 0) continue;
        // noise
        if(host_degree[i] < min_points) {
            labels[i] = -1;
            continue;
        }
        // BFS
        counter++;
        bfs(i, counter, labels);
    }
    return counter;
}


bool 
ParallelDBScanner::isEmpty(std::vector<size_t>& border)
{
    for(size_t i = 0; i < border.size(); i++) {
        if(border[i]) {
            return false;
        }
    }
    return true;
}


__global__ void
bfs_kernel(size_t* border, int* labels, int counter) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cuConstParams.num_points) return;
    if(border[j] == 0) return;  // not in the current depth of traverse
    border[j] = 0; 
    labels[j] = counter;
    if(cuConstParams.degree[j] < cuConstParams.min_points) {
        return;
    }
    size_t start_index = cuConstParams.start_index[j];
    size_t end_index = start_index + cuConstParams.degree[j];
    for(size_t neighbor_index = start_index; 
        neighbor_index < end_index; 
        neighbor_index++) {
        size_t neighbor = cuConstParams.adj_list[neighbor_index];
        if(labels[neighbor] <= 0) {
            border[neighbor] = 1;
        }
    }
}


void 
ParallelDBScanner::bfs_cuda(size_t* border, int* labels, int counter)
{
    const int blocks = CEIL(num_points, THREADS_PER_BLOCK);
    size_t* device_border;
    int* device_labels;

    int border_bytes = sizeof(size_t) * num_points;
    int labels_bytes = sizeof(int) * num_points;
    cudaMalloc(&device_border, border_bytes);
    cudaMalloc(&device_labels, labels_bytes);

    cudaMemcpy(device_border, border, border_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_labels, labels, labels_bytes, cudaMemcpyHostToDevice);
    bfs_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_border, device_labels, counter);
    cudaMemcpy(border, device_border, border_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(labels, device_labels, labels_bytes, cudaMemcpyDeviceToHost);

    cudaFree(device_border);
    cudaFree(device_labels);
}


// bfs starting at point i and current number of clusters = counter
void 
ParallelDBScanner::bfs(size_t i, size_t counter, std::vector<int> &labels)
{
    // TODO: use bit vector
    std::vector<size_t> border(num_points, 0);
    border[i] = 1;
    while(!isEmpty(border)) {
        bfs_cuda(border.data(), labels.data(), counter);
    }
}


__global__ void
degree_kernel(){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_points = cuConstParams.num_points;
    if (v >= num_points) return; 
    float2* points = (float2*)(cuConstParams.points);

    float2 p1 = points[v];
    size_t degree = 0;
    for(size_t i = 0; i < num_points; i++){
        float2 p2 = points[i];
        if(squared_distance(p1, p2) <= cuConstParams.squared_eps){
            degree++;
        }
    }
    cuConstParams.degree[v] = degree;
}


 __global__ void
adj_list_kernel()
{
    int v = blockIdx.x * blockDim.x + threadIdx.x; 
    size_t num_points = cuConstParams.num_points;

    if (v >= num_points) return;
    size_t cur_index = cuConstParams.start_index[v];
    float2* points = (float2*)(cuConstParams.points);
    float2 p1 = points[v];
    for(size_t i = 0; i < cuConstParams.num_points; i++) {
        float2 p2 = points[i];
        if(squared_distance(p1, p2) <= cuConstParams.squared_eps){
            cuConstParams.adj_list[cur_index] = i;
            cur_index++; 
        }
    }
}


void
ParallelDBScanner::construct_graph(){
    // calculate degree
    const int blocks = CEIL(num_points, THREADS_PER_BLOCK);
    degree_kernel<<<blocks, THREADS_PER_BLOCK>>>();
    cudaMemcpy(host_degree, device_degree, sizeof(size_t)*num_points,
                cudaMemcpyDeviceToHost);

    // calculate start index of each point in the adj_list
    thrust::exclusive_scan(thrust::device, 
                           device_degree, device_degree + num_points, 
                           device_start_index);
  
    // calculate adj list length and malloc adj_list
    size_t last_index, last_degree;
    cudaMemcpy(&last_degree, device_degree + num_points - 1, sizeof(size_t), 
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_index, device_start_index + num_points - 1, sizeof(size_t),
                cudaMemcpyDeviceToHost);
    adj_list_size = last_degree + last_index;
    cudaMalloc(&device_adj_list, sizeof(size_t) * adj_list_size);

    // copy over new constants
    GlobalConstants params;
    params.squared_eps = squared_eps;
    params.min_points = min_points;
    params.num_points = num_points;
    params.adj_list_size = adj_list_size;
    params.points = device_points;
    params.degree = device_degree;
    params.start_index = device_start_index;
    params.adj_list = device_adj_list;
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    // compute adjacency list
    adj_list_kernel<<<blocks, THREADS_PER_BLOCK>>>();
}

void 
ParallelDBScanner::check_scanner_const(){
    std::cout << "host: squared_eps: " << squared_eps << 
        " min_points: " << min_points << " num_points: " << num_points << 
        " adj_list_size: " << adj_list_size << std::endl;
    std::cout << "host: device_points: " << device_points <<
        ", device_degree: " << device_degree << 
        ", device_start_index: " << device_start_index <<
        ", device_adj_list: " << device_adj_list << std::endl;
    std::cout << "host_degree[0]: " << host_degree[0] <<
        " host_degree[999]: " << host_degree[999] << std::endl;
}

std::unique_ptr<DBScanner> createParallelDBScanner(){
    return std::make_unique<ParallelDBScanner>();
}


