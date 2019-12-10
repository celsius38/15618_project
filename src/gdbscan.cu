#include <memory>
#include <math.h>
#include <stddef.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "dbscan.h"
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
    float eps;
    size_t min_points;
    size_t num_points;

    float min_x;
    float min_y;
    float side;
    int row_bins;
    int col_bins;

    float* points;
    size_t* point_index;
    int* bin_index;
    size_t* bin_start_index;
    size_t* bin_end_index;
    size_t* degree;
    size_t* start_index;
    size_t adj_list_size;
    size_t* adj_list;
};

__constant__ GlobalConstants cuConstParams;

const int THREADS_PER_BLOCK = 512;


__global__ void
check_cuda_const(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i==0){
        for(int j = 0; j < 10; ++j){
            printf("%lu %lu\n", cuConstParams.bin_start_index[j],
                                cuConstParams.bin_end_index[j]);
        }
    }
}


__device__ float 
distance(float2 p1, float2 p2){
    float x = p1.x - p2.x;
    float y = p1.y - p2.y;
    return sqrt(x*x + y*y);
}

class GDBScanner: public DBScanner
{
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    GDBScanner();
    ~GDBScanner();
    size_t scan(
        std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t min_points
    );

private:
    float eps;
    size_t min_points;
    size_t num_points;

    float min_x;
    float min_y; 
    float side;
    int row_bins;
    int col_bins;

    float* device_points;
    size_t* device_point_index;
    int* device_bin_index;
    size_t* device_bin_start_index;
    size_t* device_bin_end_index;
    size_t* host_degree;
    size_t* device_degree;
    size_t* device_start_index;
    size_t adj_list_size;
    size_t* device_adj_list;

    void setup(std::vector<Vec2> &points, float eps, size_t min_points);
    
    bool isEmpty(std::vector<size_t>& border);

    void bfs(size_t i, size_t counter, std::vector<int>& labels);

    void construct_graph();

    void bfs_cuda(size_t* border, int* labels, int counter);

    void check_scanner_const();
};

GDBScanner::GDBScanner(){
    eps = 0.f;
    min_points = 0;
    num_points = 0;

    min_x = 0.f;
    min_y = 0.f;
    row_bins = 0;
    col_bins = 0;

    device_points = NULL;
    device_point_index = NULL;
    device_bin_index = NULL;
    device_bin_start_index = NULL;
    device_bin_end_index = NULL;
    host_degree = NULL;
    device_degree = NULL;
    device_start_index = NULL;
    adj_list_size = 0;
    device_adj_list = NULL;
}


GDBScanner::~GDBScanner(){
    if(host_degree) delete[] host_degree;

    if(device_points) cudaFree(device_points);
    if(device_degree) cudaFree(device_degree);
    if(device_start_index) cudaFree(device_start_index);
    if(device_adj_list) cudaFree(device_adj_list);
}


void
GDBScanner::setup(std::vector<Vec2> &points, float eps, size_t min_points)
{
    // allocate host data
    this -> eps = eps;
    this -> min_points = min_points;
    this -> num_points = points.size();
    this -> host_degree = new size_t[points.size()];

    // find lower left point and row_bins, col_bins
    float min_x(0.), min_y(0.);
    float max_x(0.), max_y(0.);
    for(size_t i = 0; i < points.size(); ++i){
        Vec2 point = points[i];
        min_x = MIN(min_x, point.x);
        min_y = MIN(min_y, point.y);
        max_x = MAX(max_x, point.x);
        max_y = MAX(max_y, point.y);
    }
    float side = eps * 1.01f;
    row_bins = ceil((max_y - min_y)/side);
    col_bins = ceil((max_x - min_x)/side);
    this -> min_y = min_y;
    this -> min_x = min_x;

    // allocate device data
    size_t points_bytes = sizeof(float) * 2 * num_points;
    size_t bin_index_bytes = sizeof(int) * num_points;
    size_t bin_start_index_bytes = sizeof(size_t) * row_bins * col_bins;
    size_t bytes = sizeof(size_t) * num_points;
    cudaMalloc(&device_points, points_bytes);
    cudaMalloc(&device_point_index, bytes);
    cudaMalloc(&device_bin_index, bin_index_bytes);
    cudaMalloc(&device_bin_start_index, bin_start_index_bytes);
    cudaMalloc(&device_bin_end_index, bin_start_index_bytes);
    cudaMemset(device_bin_start_index, 0, bin_start_index_bytes);
    cudaMemset(device_bin_end_index, 0, bin_start_index_bytes);
    cudaMalloc(&device_degree, bytes);
    cudaMalloc(&device_start_index, bytes);
   
    // copy data and hyper parameters to device 
    cudaMemcpy(device_points, (float*)points.data(), points_bytes, cudaMemcpyHostToDevice); 
    // copy hyper parameters to device
    GlobalConstants params;
    params.eps = eps;
    params.min_points = min_points;
    params.num_points = num_points;
    params.min_x = min_x;
    params.min_y = min_y;
    params.row_bins = row_bins;
    params.col_bins = col_bins;
    params.side = side;
    params.points = device_points;
    params.point_index = device_point_index;
    params.bin_index = device_bin_index;
    params.bin_start_index = device_bin_start_index;
    params.bin_end_index = device_bin_end_index;
    params.degree = device_degree;
    params.start_index = device_start_index; 
    params.adj_list_size = 0;
    params.adj_list = NULL;
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));
}


size_t 
GDBScanner::scan(
    std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t min_points
)
{
    setup(points, eps, min_points);
    construct_graph();

    size_t counter = 0;  // current number of clusters
    for(size_t i = 0; i < points.size(); i++){
#if defined(DEBUG)
        std::cout << i << ": " << host_degree[i] << std::endl; 
#endif
        // already in a cluster, skip
        if(labels[i] > 0) continue;
        // noise
        if(host_degree[i] < min_points) {
            labels[i] = -1;
            continue;
        }
        // BFS
        ++counter;
        bfs(i, counter, labels);
    }
    return counter;
}


bool 
GDBScanner::isEmpty(std::vector<size_t>& border)
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
GDBScanner::bfs_cuda(size_t* border, int* labels, int counter)
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
GDBScanner::bfs(size_t i, size_t counter, std::vector<int> &labels)
{
    // TODO: use bit vector
    std::vector<size_t> border(num_points, 0);
    border[i] = 1;
    while(!isEmpty(border)) {
        bfs_cuda(border.data(), labels.data(), counter);
    }
}


__device__ size_t
degree_in_bin(float2 p1, int row_idx, int col_idx){
    if( (row_idx < 0) || (row_idx >= cuConstParams.row_bins) ||
        (col_idx < 0) || (col_idx >= cuConstParams.col_bins))
        return 0;
    int bin_idx = row_idx * cuConstParams.col_bins + col_idx;
    size_t bin_start = cuConstParams.bin_start_index[bin_idx];
    size_t bin_end = cuConstParams.bin_end_index[bin_idx];
    size_t degree = 0;
    for(size_t p = bin_start; p < bin_end; ++p){
        size_t pid = cuConstParams.point_index[p];
        float2 p2 = ((float2*)(cuConstParams.points))[pid];
        if(distance(p1, p2) <= cuConstParams.eps) ++degree;
    }
    return degree;
}

__global__ void
degree_kernel(){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= cuConstParams.num_points) return; 
    size_t pid = cuConstParams.point_index[v];
    float2 p1 = ((float2*)(cuConstParams.points))[pid];
    int bin_idx = cuConstParams.bin_index[v];
    int col_bins = cuConstParams.col_bins;
    int row_idx = bin_idx/col_bins;
    int col_idx = bin_idx % col_bins;
    size_t degree = 0;
    for(int row_diff = -1; row_diff <2; ++row_diff){
        for(int col_diff = -1; col_diff < 2; ++col_diff){
            degree += degree_in_bin(p1, row_idx + row_diff, col_idx + col_diff);
        }
    } 
    cuConstParams.degree[pid] = degree;
}


__device__ size_t
neighbor_in_bin(float2 p1, int row_idx, int col_idx, size_t idx)
{
    if( (row_idx < 0) || (row_idx >= cuConstParams.row_bins) ||
        (col_idx < 0) || (col_idx >= cuConstParams.col_bins))
        return idx;
    int bin_idx = row_idx * cuConstParams.col_bins + col_idx;
    size_t bin_start = cuConstParams.bin_start_index[bin_idx];
    size_t bin_end = cuConstParams.bin_end_index[bin_idx];
    for(size_t p = bin_start; p < bin_end; ++p){
        size_t pid = cuConstParams.point_index[p];
        float2 p2 = ((float2*)(cuConstParams.points))[pid];
        if(distance(p1, p2) <= cuConstParams.eps)
            cuConstParams.adj_list[idx++] = pid;
    }
    return idx;
}

 __global__ void
adj_list_kernel()
{
    int v = blockIdx.x * blockDim.x + threadIdx.x; 
    if (v >= cuConstParams.num_points) return;
    size_t pid = cuConstParams.point_index[v];
    float2 p1 = ((float2*)(cuConstParams.points))[pid];
    int bin_idx = cuConstParams.bin_index[v];
    size_t cur_idx = cuConstParams.start_index[pid];
    int row_idx = bin_idx/cuConstParams.col_bins;
    int col_idx = bin_idx%cuConstParams.col_bins;
    for(int row_diff = -1; row_diff < 2; ++row_diff){
        for(int col_diff = -1; col_diff < 2; ++col_diff){
            cur_idx = neighbor_in_bin(p1, row_idx + row_diff, col_idx + col_diff,
                                      cur_idx);
        }
    }
}


__global__ void
binning_kernel()
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= cuConstParams.num_points) return;
    float2 point = ((float2*)(cuConstParams.points))[v];
    float side = cuConstParams.side;
    int col_idx = (point.x - cuConstParams.min_x)/side;
    int row_idx = (point.y - cuConstParams.min_y)/side;
    cuConstParams.bin_index[v] = row_idx * cuConstParams.col_bins + col_idx;
    cuConstParams.point_index[v] = v;
}


__global__ void
find_bin_start_kernel()
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= cuConstParams.num_points) return;
    int bin_idx = cuConstParams.bin_index[v];
    if(v == 0){
        cuConstParams.bin_start_index[bin_idx] = 0;
    }else{
        int last_bin_idx = cuConstParams.bin_index[v-1];
        if(bin_idx != last_bin_idx){
            cuConstParams.bin_start_index[bin_idx] = v;
            cuConstParams.bin_end_index[last_bin_idx] = v;
        }
    }
    if(v == cuConstParams.num_points - 1){
        cuConstParams.bin_end_index[bin_idx] = cuConstParams.num_points;
    }
}


void
GDBScanner::construct_graph(){
    // binning
    const int blocks = CEIL(num_points, THREADS_PER_BLOCK);
    binning_kernel<<<blocks, THREADS_PER_BLOCK>>>();
    thrust::sort_by_key(thrust::device, 
                        device_bin_index, device_bin_index + num_points, 
                        device_point_index); 
    find_bin_start_kernel<<<blocks,THREADS_PER_BLOCK>>>();

    // calculate degree
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
    std::cout << "adj list size: " << adj_list_size << std::endl;

    // copy over new constants
    GlobalConstants params;
    params.eps = eps;
    params.min_points = min_points;
    params.num_points = num_points;
    params.min_x = min_x;
    params.min_y = min_y;
    params.row_bins = row_bins;
    params.col_bins = col_bins;
    params.side = side;
    params.points = device_points;
    params.point_index = device_point_index;
    params.bin_index = device_bin_index;
    params.bin_start_index = device_bin_start_index;
    params.bin_end_index = device_bin_end_index;
    params.degree = device_degree;
    params.start_index = device_start_index; 
    params.adj_list_size = adj_list_size;
    params.adj_list = device_adj_list;
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    // compute adjacency list
    adj_list_kernel<<<blocks, THREADS_PER_BLOCK>>>();
    cudaDeviceSynchronize();
}

void 
GDBScanner::check_scanner_const(){
    std::cout << "host: eps: " << eps << 
        " min_points: " << min_points << " num_points: " << num_points << 
        " adj_list_size: " << adj_list_size << std::endl;
    std::cout << "host: device_points: " << device_points <<
        ", device_degree: " << device_degree << 
        ", device_start_index: " << device_start_index <<
        ", device_adj_list: " << device_adj_list << std::endl;
    std::cout << "host_degree[0]: " << host_degree[0] <<
        " host_degree[999]: " << host_degree[999] << std::endl;
}

std::unique_ptr<DBScanner> createGDBScanner()
{
    return std::make_unique<GDBScanner>();
}


