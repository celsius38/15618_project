#include "dbscan.h"
#include "mpi.h"
#include "make_unique.h"
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>

#define MASTER 0

using namespace std;
MPI_Datatype MPI_Cell;

class Cell {
public:
    short is_core; // 1: core cell, 0: non-core cell
    size_t id;
    size_t degree;
    size_t start_index;
};

class RPDBScanner: public DBScanner {
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    size_t scan(std::vector<Vec2> &points, 
                std::vector<int> &labels, 
                float eps, 
                size_t minPts) {
        // other workers' cluster count will not be changed
        // indicate main function its identity
        // only master will print out messages
        int cluster_count = 0;
        int numtasks, taskid;
        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
        MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
        createMPICell();

        // Stage 1: data partition

        // Stage 2: build local clustering

        // Stage 3: merge clustering

        // TODO: partition assigned, calculated by Stage 2
        size_t local_adj_list_len;
        vector<size_t> local_adj_list;
        size_t local_cell_count;
        vector<Cell> local_partition;

        size_t local_point_count;
        vector<size_t> local_point_id;
        vector<short> local_point_is_core;

        if(taskid != MASTER) {
            // send cell graph
            MPI_Send(&local_adj_list_len, 1, MPI_UNSIGNED_LONG, MASTER, 1, MPI_COMM_WORLD);
            MPI_Send(&local_adj_list[0], local_adj_list_len, MPI_UNSIGNED_LONG, MASTER, 2, MPI_COMM_WORLD);
            MPI_Send(&local_cell_count, 1, MPI_UNSIGNED_LONG, MASTER, 3, MPI_COMM_WORLD);
            MPI_Send(&local_partition[0], local_cell_count, MPI_Cell, MASTER, 4, MPI_COMM_WORLD);

            // send point_is_core
            MPI_Send(&local_point_count, 1, MPI_UNSIGNED_LONG, MASTER, 5, MPI_COMM_WORLD);
            MPI_Send(&local_point_id[0], local_point_count, MPI_UNSIGNED_LONG, MASTER, 6, MPI_COMM_WORLD);
            MPI_Send(&local_point_is_core[0], local_point_count, MPI_SHORT, MASTER, 7, MPI_COMM_WORLD);
        }

        if(taskid == MASTER) {
            // key is point id, value is is_core
            unordered_map<size_t, short> point_is_core_map; 
            addIntoMap(point_is_core_map, local_point_id, local_point_is_core);
            // TODO: tree like merge
            for(int i = 1; i < numtasks; i++) {
                // receive a cell graph
                size_t other_adj_list_len;
                vector<size_t> other_adj_list;
                size_t other_cell_count;
                vector<Cell> other_partition;

                MPI_Status status;
                MPI_Recv(&other_adj_list_len, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                other_adj_list.resize(other_adj_list_len);
                MPI_Recv(&other_adj_list[0], other_adj_list_len, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&other_cell_count, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
                other_partition.resize(other_cell_count);
                MPI_Recv(&other_partition[0], other_cell_count, MPI_Cell, status.MPI_SOURCE, 4, MPI_COMM_WORLD, &status);

                // merge two cell graphs at a time
                vector<Cell> merged_partition;
                vector<size_t> merged_adj_list;
                mergeTwoGraph(local_partition, local_adj_list, 
                    other_partition, other_adj_list, 
                    merged_partition, merged_adj_list);
                local_partition = merged_partition;
                local_adj_list = merged_adj_list;

                // receive point_is_core
                size_t other_point_count;
                vector<size_t> other_point_id;
                vector<short> other_point_is_core;

                MPI_Recv(&other_point_count, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 5, MPI_COMM_WORLD, &status);
                other_point_id.resize(other_point_count);
                other_point_is_core.resize(other_point_count);
                MPI_Recv(&other_point_id[0], other_point_count, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 6, MPI_COMM_WORLD, &status);
                MPI_Recv(&other_point_is_core[0], other_point_count, MPI_SHORT, status.MPI_SOURCE, 7, MPI_COMM_WORLD, &status);

                // TODO add into point_is_core_map
                addIntoMap(point_is_core_map, other_point_id, other_point_is_core);
            }
            // local_partition and local_adj_list for master now is the global graph
            vector<int> cell_cluster_id(local_partition.size(), 0);
            cluster_count = labelCoreCells(local_partition, local_adj_list, cell_cluster_id);
            labelPointsInCoreCells(cell_cluster_id, labels);
            // TODO label points in non core cells
            // TODO combine and get global point_is_core
            labelPointsInNonCoreCells(cell_cluster_id, point_is_core_map, labels);
        }

        MPI_Type_free(&MPI_Cell);
        MPI_Finalize();
        return cluster_count;
    }

private:
    // TODO: real
    size_t point_index[10] = {1,8,5,7,2,3,4,9,0,6}; // len = number of points
    size_t bin_index[10] = {0,0,1,1,2,2,3,3,4}; // len = number of points
    size_t bin_start_index[5] = {0,2,4,6,8}; // len = number of bins
    size_t bin_end_index[5] = {2,4,6,8,10}; // len = number of bins
    int col_bins;
    int row_bins;
    vector<Vec2> points;
    float eps;

    void createMPICell() {
        MPI_Datatype oldtypes[2] = {MPI_SHORT, MPI_UNSIGNED_LONG};
        int blockcounts[2];
        MPI_Aint offsets[2];
        // 1 block of short starting at 0
        offsets[0] = 0;
        blockcounts[0] = 1;
        // 3 blocks of size_t starting at `extent`
        MPI_Aint extent;
        MPI_Aint lb;
        MPI_Type_get_extent(MPI_SHORT, &lb, &extent);
        offsets[1] = extent;
        blockcounts[1] = 3;
        MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &MPI_Cell);
        MPI_Type_commit(&MPI_Cell);
    }

    void addIntoMap(unordered_map<size_t, short> point_is_core_map, vector<size_t> point_id, vector<short> point_is_core) {
        for(int i = 0; i < point_id.size(); i++) {
            point_is_core_map[point_id[i]] = point_is_core[i];
        }
    }

    // add a cell into the merged graph
    void addCell(Cell cell, vector<Cell>& all_cells, vector<size_t>& adj_list_sub, vector<size_t>& adj_list) {
        Cell new_cell;
        new_cell.id = cell.id;
        new_cell.degree = cell.degree;
        new_cell.is_core = cell.is_core;
        if(all_cells.size() == 0) {
            new_cell.start_index = 0;
        } else {
            new_cell.start_index = all_cells[all_cells.size()-1].start_index + all_cells[all_cells.size()-1].degree;
        }
        all_cells.push_back(new_cell);
        
        for(size_t neighbour_index = cell.start_index; 
            neighbour_index < cell.start_index+cell.degree; 
            neighbour_index++) {
            adj_list.push_back(adj_list_sub[neighbour_index]);
        }
    }

    // merge two partions' cell garphs
    // store the merged graph info in "all_cells" and "adj_list"
    void mergeTwoGraph(vector<Cell>& partition1, vector<size_t>& adj_list1, 
                  vector<Cell>& partition2, vector<size_t>& adj_list2, 
                  vector<Cell>& all_cells, vector<size_t>& adj_list) {
        int ptr1 = 0;
        int ptr2 = 0;
        while(ptr1 < partition1.size() && ptr2 < partition2.size()) {
            if(partition1[ptr1].id < partition2[ptr2].id) {
                addCell(partition1[ptr1], all_cells, adj_list1, adj_list);
                ptr1++;
            } else {
                addCell(partition2[ptr2], all_cells, adj_list2, adj_list);
                ptr2++;
            }
        }
        while(ptr1 < partition1.size()) {
            addCell(partition1[ptr1], all_cells, adj_list1, adj_list);
            ptr1++;
        }
        while(ptr2 < partition2.size()) {
            addCell(partition2[ptr2], all_cells, adj_list2, adj_list);
            ptr2++;
        }
    }

    // label all core cells
    int labelCoreCells(vector<Cell>& all_cells, vector<size_t>& adj_list, vector<int>& cell_cluster_id) {
        int cluster_id = 0;
        vector<int> visited(all_cells.size(), 0);
        for(int cell_id = 0; cell_id < all_cells.size(); cell_id++) {
            if(visited[cell_id] || all_cells[cell_id].is_core == 0) {
                continue;
            }
            // the first cluster id is 1
            cluster_id++;
            bfs(cell_id, visited, all_cells, adj_list, cluster_id, cell_cluster_id);
        }
        return cluster_id;
    }

    // TODO: CUDA parallel
    // label all connected core cells the same cluster
    void bfs(int root_cell_id, 
             vector<int>& visited, 
             vector<Cell> all_cells,
             vector<size_t>& adj_list, 
             int cluster_id,
             vector<int>& cell_cluster_id) {
        queue<int> queue;
        queue.push(root_cell_id);
        visited[root_cell_id] = 1;
        cell_cluster_id[root_cell_id] = cluster_id;
        while(!queue.empty()) {
            int cell_id = queue.front();
            queue.pop();
            for(int neighbour_index = all_cells[cell_id].start_index; 
                neighbour_index < all_cells[cell_id].start_index+all_cells[cell_id].degree;
                neighbour_index++) {
                int neighbour = adj_list[neighbour_index];
                if(!visited[neighbour] && all_cells[neighbour].is_core == 1) {
                    queue.push(neighbour);
                    visited[neighbour] = 1;
                    cell_cluster_id[neighbour] = cluster_id;
                }
            }
        }
    }

    void labelPointsInCoreCells(vector<int>& cell_cluster_id, vector<int>& labels) {
        for(int cell_id = 0; cell_id < cell_cluster_id.size(); cell_id++) {
            if(cell_cluster_id[cell_id] > 0) {
                // label one core cell
                for(size_t i = bin_start_index[cell_id]; i < bin_end_index[cell_id]; i++) {
                    size_t point_id = point_index[i];
                    labels[point_id] = cell_cluster_id[cell_id];
                }
            }
        }
    }

    void labelPointsInNonCoreCells(vector<int>& cell_cluster_id, 
                                    unordered_map<size_t, short> point_is_core_map,
                                    vector<int>& labels) {
        for(int cell_id = 0; cell_id < cell_cluster_id.size(); cell_id++) {
            if(cell_cluster_id[cell_id] > 0) {
                // label one non core cell
                for(size_t i = bin_start_index[cell_id]; i < bin_end_index[cell_id]; i++) {
                    size_t point_id = point_index[i];
                    // label one point
                    // TODO
                    vector<size_t> neighbours = findNeighbours(point_id, cell_id);
                    for(size_t neighbour_index = 0; neighbour_index < neighbours.size(); neighbour_index++) {
                        size_t neighbour_id = neighbours[neighbour_index];
                        // neighbour is core
                        if(point_is_core_map[neighbour_id] == 1) {
                            labels[point_id] = labels[neighbour_id];
                            break;
                        }
                    }
                }
            }
        }
    }
    
    vector<size_t> findNeighbours(size_t point_id, int cell_id) {
        vector<size_t> neighbours;
        int cell_row_id = cell_id/col_bins;
        int cell_col_id = cell_id%col_bins;
        for(int row_diff = -2; row_diff < 3; ++row_diff) {
            for(int col_diff = -2; col_diff < 3; ++col_diff) {
                addOneCellNeighbours(neighbours, point_id, cell_row_id+row_diff, cell_col_id+col_diff);
            }
        }
    }

    void addOneCellNeighbours(vector<size_t> neighbours, size_t point_id, int cell_row_id, int cell_col_id) {
        if(cell_row_id < 0 || cell_row_id > row_bins || cell_col_id < 0 || cell_col_id > col_bins) {
            return;
        }
        int cell_id = cell_row_id * col_bins + cell_col_id;
        for(size_t i = bin_start_index[cell_id]; i < bin_end_index[cell_id]; i++) {
            size_t other_point_id = point_index[i];
            if((points[other_point_id]-points[point_id]).length() <= eps) {
                neighbours.push_back(other_point_id);
            }
        }
    }
};

std::unique_ptr<DBScanner> createRPDBScanner(){
    return std::make_unique<RPDBScanner>();
}
