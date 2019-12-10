#include <stdlib.h>
#include <memory>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include "dbscan.h"

__constant__ HyperParameters constParams;
const int THREADS_PER_BLOCK = 512;

void
setup_device(HyperParameters* params, std::vector<Vec2> &points) {
    // allocate device data
    cudaMalloc(&params->points, sizeof(float)*2*params->num_points);
    cudaMalloc(&params->point_index, sizeof(size_t)*params->num_points);
    cudaMalloc(&params->cell_index, sizeof(size_t)*params->num_points);
    cudaMalloc(&params->cell_start_index, sizeof(size_t)*params->num_cells);
    cudaMalloc(&params->cell_end_index, sizeof(size_t)*params->num_cells);
    cudaMemset(params->cell_start_index, 0, sizeof(size_t)*params->num_cells);
    cudaMemset(params->cell_end_index, 0, sizeof(size_t)*params->num_cells);

    // copy points to device
    cudaMemcpy(params->points, (float*)points.data(), sizeof(float)*2*params->num_points, cudaMemcpyHostToDevice);

    // copy constant to device
    cudaMemcpyToSymbol(constParams, params, sizeof(HyperParameters));
}


__global__ void
assign_cell_id_kernel()
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if( v >= constParams.num_points) return;
    float2 point = ((float2*)(constParams.points))[v];
    float side = constParams.side;
    int col_idx = (point.x - constParams.min_x)/side;
    int row_idx = (point.y - constParams.min_y)/side;
    constParams.cell_index[v] = row_idx * constParams.col_cells + col_idx;
    constParams.point_index[v] = v;
} 


// assume now that point_index is sorted by corresponding cell_index
__global__ void
find_cell_start_end_kernel()
{ 
    int v = blockIdx.x * blockDim.x + threadIdx.x; 
    if( v >= constParams.num_points) return;
    int cell_idx = constParams.cell_index[v];
    if( v == 0 ){
        constParams.cell_start_index[cell_idx] = 0;
    }else{
        int last_cell_idx = constParams.cell_index[v-1];
        if(cell_idx != last_cell_idx){
            constParams.cell_start_index[cell_idx] = v;
            constParams.cell_end_index[last_cell_idx] = v;
        }
    }
    if( v == constParams.num_points - 1){// last point
        constParams.cell_end_index[cell_idx] = constParams.num_points;
    }
} 

void
construct_global_graph(HyperParameters* params, 
    size_t* host_point_index, size_t* host_cell_index, size_t* host_cell_start_index, size_t* host_cell_end_index) { 
    // assign cell id and cell start end
    const int blocks = CEIL(params->num_points, THREADS_PER_BLOCK);
    assign_cell_id_kernel<<<blocks, THREADS_PER_BLOCK>>>();
    thrust::sort_by_key(thrust::device,
                        params->cell_index, params->cell_index + params->num_points,
                        params->point_index);
    find_cell_start_end_kernel<<<blocks, THREADS_PER_BLOCK>>>();

    // copy global graph back to host
    cudaMemcpy(host_point_index, params->point_index, sizeof(size_t)*params->num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cell_index, params->cell_index, sizeof(size_t)*params->num_points, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cell_start_index, params->cell_start_index, sizeof(size_t)*params->num_cells, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_cell_end_index, params->cell_end_index, sizeof(size_t)*params->num_cells, cudaMemcpyDeviceToHost);
}


/*
// given a point p1, calculate the number of neighbors in 
// cell(row_idx, col_idx)
__device__ size_t
point_degree_in_cell(float2 p1, int row_idx, int col_idx){
    if( (row_idx < 0) || (row_idx >= constParams.row_cells) ||
        (col_idx < 0) || (col_idx >= constParams.col_cells))
        return 0;
    int cell_idx = row_idx * constParams.col_cells + col_idx;
    size_t cell_start = constParams.cell_start_index[cell_idx];
    size_t cell_end = constParams.cell_end_index[cell_idx];
    size_t degree = 0;
    for(size_t p = cell_start; p < cell_end; ++p){
        size_t pid = constParams.point_index[p];
        float2 p2 = ((float2*)(constParams.points))[pid];
        if(distance(p1, p2) <= constParams.eps) ++degree;
    }
    return degree;
}


__global__ void
mark_core_kernel(int partition_size, int num_points_in_partition,
                 size_t* raw_index, int* cell_order,
                 size_t* point_degree, short* point_is_core,
                 short* cell_is_core)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= num_points_in_partition) return;
    size_t raw_idx = raw_index[v];
    size_t pid = constParams.point_index[raw_idx];
    size_t cid = constParams.cell_index[raw_idx];
    int row_idx = cid/constParams.col_cells;
    int col_idx = cid % constParams.col_cells;
    float2 p1 =((float2*)constParams.points)[pid];
    size_t degree = 0;
    for(int row_diff = -2; row_diff < 3; ++row_diff){
        for(int col_diff = -2; col_diff < 3; ++col_diff){
            degree += point_degree_in_cell(p1, row_idx + row_diff, col_idx + col+diff);
        }
    }
    point_degree[v] = degree;
    short is_core = (degree >= constParams.min_points ? 1 : 0);
    if(is_core){
        point_is_core[v] = is_core;
        cell_is_core[cell_order[v]] = is_core;
    }
}


// given a core point p1, tell if cell(row_idx, col_idx) is reachable
// from the cell p1 is in
__device__ void
mark_cell_to_cell_edge(float2 p1, int row_idx, int col_idx,
                        short* cell_to_cell_edge,
                        int source_cell_order, int target_cell_order)
{
    if( (row_idx < 0) || (row_idx >= constParams.row_cells) ||
        (col_idx < 0) || (col_idx >= constParams.col_cells))
        return;
    int cell_idx = row_idx * constParams.col_cells + col_idx;
    size_t cell_start = constParams.cell_start_index[cell_idx];
    size_t cell_end = constParams.cell_end_index[cell_idx];
    for(size_t p = cell_start; p < cell_end; ++p){
        size_t pid = constParams.point_index[p];
        float2 p2 = ((float2*)(constParams.points))[pid]; 
        if(distance(p1, p2) <= constParams.eps){ 
            cell_to_cell_edge[source_cell_order*25+target_cell_order] = 1;
            return;
        }
    }
}


__global__ void
cell_to_cell_edge_kernel(size_t num_points_in_partition,
                         size_t* raw_index, int* cell_order,
                         size_t* point_degree, short* point_is_core,
                         short* cell_is_core, short* cell_to_cell_edge)
{ 
    int v = blockIdx.x * blockDim.x + threadIdx.x; 
    if(v >= num_points_in_partition) return;
    if(point_degree[v] < constParams.min_points) return;
    size_t raw_idx = raw_index[v]; 
    size_t pid = constParams.point_index[raw_index];
    size_t cid = constParams.cell_index[raw_index];
    size_t row_idx = cid/constParams.col_cells;
    size_t col_idx = cid%constParams.col_cells;
    float2 p1 = ((float2*)constParams.points)[pid];
    for(int row_diff = -2; row_diff < 3; ++row_diff){
        for(int col_diff = -2; col_diff < 3; ++col_diff){
            if((row_diff == 0) && (col_diff == 0)) continue;
            mark_cell_to_cell_edge(p1, row_idx+row_diff, col_idx+col_diff,
                                    cell_to_cell_edge,
                                    cell_order[v], (row_diff+2)*5+(col_diff+2));
        }
    }
}

// each partition is a list of cell id 

void
construct_partial_cell_graph(
    vector<int> partition,
    std::vector<size_t>& ret_point_ids, std::vector<short>& ret_point_is_core,
    std::vector<Cell>& ret_cells, std::vector<size_t>& ret_cell_adj_list
)
{
    // calculate num_points in partition 
    size_t partition_size = partition.size();
    vector<size_t> raw_index;  // raw idx into point_index and cell_index
    vector<int> cell_order; // cell order in partition for each raw idx
    int order = 0;
    for(int i = 0; i < partition_size; ++i){
        int cell_id = partition[i];
        for(size_t j = cell_start_index[cell_id]; j < cell_end_index[cell_id]; ++j){
            raw_index.push_back(j);
            cell_order.push_back(order);
        }
        ++order;
    }
    size_t num_points_in_partition = raw_index.size();
    
    // allocate device data
    size_t* device_raw_index;
    size_t* device_cell_order;
    size_t* device_point_degree;
    short* device_point_is_core;
    short* device_cell_is_core;
    short* device_cell_to_cell_edge;
    cudaMalloc(&device_raw_index, sizeof(size_t)*num_points_in_partition);
    cudaMalloc(&device_cell_order, sizeof(size_t)*num_points_in_partition);
    cudaMalloc(&device_point_degree, sizeof(size_t)*num_points_in_partition);
    cudaMalloc(&device_point_is_core, sizeof(short)*num_points_in_partition);
    cudaMalloc(&device_cell_is_core, sizeof(short)*partition_size);
    cudaMalloc(&device_cell_to_cell_edge, 
                sizeof(short)*partition_size*25);
    cudaMemset(device_point_is_core, 0, sizeof(size_t)*num_points_in_partition);
    cudaMemset(device_cell_is_core, 0, sizeof(size_t)*partition_size);
    cudaMemset(device_cell_to_cell_edge, 0, sizeof(short)*partition_size*25);

    // copy host data to device
    cudaMemcpy(device_raw_index, raw_index, sizeof(size_t)*num_points_in_partition,
                cudaMemcpyHostToDevice);
    cudaMemcpy(device_cell_order, cell_order, sizeof(int)*num_points_in_partition,
                cudaMemcpyHostToDevice);

    // calculate if points and cells in partition are cores
    const int blocks = CEIL(num_points_in_partition, THREADS_PER_BLOCK);
    mark_core_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        partition_size, num_points_in_partition, device_raw_index,
        device_cell_order, device_point_degree, device_point_is_core,
        device_cell_is_core); 
    cell_to_cell_edge_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            num_points_in_partition,
            device_raw_index, device_cell_order,
            device_point_degree, device_point_is_core,
            device_cell_is_core, device_cell_to_cell_edge
    );

    // fill the return result
    for(size_t i = 0; i < num_points_in_partition){
        point_ids.push_back(point_index[raw_index[i]]);
    }
    ret_point_is_core.insert(ret_point_is_core.end(), point_is_core, point_is_core + num_points_in_partition);
    size_t start_index = 0;
    for(size_t i = 0; i < partition_size; ++i){
        size_t degree = 0;
        size_t cid = cell_index[raw_index[i]]; 
        size_t row_idx = cid/col_cells;
        size_t col_idx = cid%col_cells;
        for(int row_diff = -2; row_diff < 3; ++row_diff){
            for(int col_diff = -2; col_diff < 3; ++col_diff){
                int tgt_row_idx = row_idx + row_diff;
                int tgt_col_idx = col_idx + col_diff;
                if( (tgt_row_idx < 0) || (tgt_row_idx > row_cells) ||
                    (tgt_col_idx < 0) || (tgt_col_idx > col_cells) )
                    continue;
                int tgt_cell_id = tgt_row_idx * col_cells + col_idx;
                int tgt_cell_order = (row_diff + 2) * 5 + col_diff;
                if(cell_to_cell_edge[i*25 + tgt_cell_order]){
                    degree += 1;
                    ret_cell_adj_list.push_back(tgt_cell_id);
                }
            }
        }
        Cell cell;
        cell.is_core = cell_is_core[i];
        cell.id = cid;
        cell.degree = degree;
        cell.start_index = start_index;
        start_index += degree;
    }
    return;

}
*/



