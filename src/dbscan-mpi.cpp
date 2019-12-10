#include "dbscan.h"
#include "mpi.h"
#include "utils.h"
#include "make_unique.h"
#include <memory>
#include <vector>
#include <queue>
#include <unordered_map>

#define MASTER 0

const unsigned int RandSeed = 15618;

MPI_Datatype MPI_Cell;

class Cell {
public:
    short is_core; // 1: core cell, 0: non-core cell
    size_t id;
    size_t degree;
    size_t start_index;
};


// TODO
/*
void
construct_partial_cell_graph(
    vector<int> partition,
    std::vector<size_t>& ret_point_ids, std::vector<short>& ret_point_is_core,
    std::vector<Cell>& ret_cells, std::vector<size_t>& ret_cell_adj_list
);
*/
void construct_global_graph(HyperParameters* params, size_t* point_index, size_t* cell_index, size_t* cell_start_index, size_t* cell_end_index);
void setup_device(HyperParameters* params, std::vector<Vec2> &points);

class RPDBScanner: public DBScanner {
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    RPDBScanner() { 
        params.eps = 0.f;
        params.min_points = 0;
        params.num_points = 0;

        params.min_x = 0.f;
        params.min_y = 0.f;
        params.side = 0.f;
        params.num_cells = 0;
        params.row_cells = 0;
        params.col_cells = 0;

        params.points = NULL;
        params.point_index = NULL;
        params.cell_index = NULL;
        params.cell_start_index = NULL;
        params.cell_end_index = NULL;
    }

    ~RPDBScanner() {
        if(params.points) delete[] params.points;
        if(params.point_index) delete[] params.point_index;
        if(params.cell_index) delete[] params.cell_index;
        if(params.cell_start_index) delete[] params.cell_start_index;
        if(params.cell_end_index) delete[] params.cell_end_index;
    }

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
        setup(points, eps, minPts);
        construct_global_graph(&params, point_index, cell_index, cell_start_index, cell_end_index);
        std::vector<int> local_cell_index = random_split(numtasks, taskid);
        size_t local_cell_count = local_cell_index.size();
        for(size_t i = 0; i < 10; ++i){
            std::cout << cell_start_index[i] << " " << cell_end_index[i] << std::endl;
        }
        std::cout << "stage 1 finished" << std::endl;

        // Stage 2: build local clustering
        std::vector<size_t> local_point_id;
        std::vector<short> local_point_is_core;
        std::vector<size_t> local_adj_list;
        std::vector<Cell> local_partition;

        //TODO: construct_partial_cell_graph(
            //local_cell_index, 
            //local_point_id, local_point_is_core,
            //local_partition, local_adj_list);
        size_t local_point_count = local_point_id.size();
        size_t local_adj_list_len = local_adj_list.size(); 
        std::cout << "stage 2 finished" << std::endl;

        // Stage 3: merge clustering
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
            std::unordered_map<size_t, short> point_is_core_map; 
            addIntoMap(point_is_core_map, local_point_id, local_point_is_core);
            // TODO: tree like merge
            for(int i = 1; i < numtasks; i++) {
                // receive a cell graph
                size_t other_adj_list_len;
                std::vector<size_t> other_adj_list;
                size_t other_cell_count;
                std::vector<Cell> other_partition;

                MPI_Status status;
                MPI_Recv(&other_adj_list_len, 1, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                other_adj_list.resize(other_adj_list_len);
                MPI_Recv(&other_adj_list[0], other_adj_list_len, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&other_cell_count, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
                other_partition.resize(other_cell_count);
                MPI_Recv(&other_partition[0], other_cell_count, MPI_Cell, status.MPI_SOURCE, 4, MPI_COMM_WORLD, &status);

                // merge two cell graphs at a time
                std::vector<Cell> merged_partition;
                std::vector<size_t> merged_adj_list;
                mergeTwoGraph(local_partition, local_adj_list, 
                    other_partition, other_adj_list, 
                    merged_partition, merged_adj_list);
                local_partition = merged_partition;
                local_adj_list = merged_adj_list;

                // receive point_is_core
                size_t other_point_count;
                std::vector<size_t> other_point_id;
                std::vector<short> other_point_is_core;

                MPI_Recv(&other_point_count, 1, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 5, MPI_COMM_WORLD, &status);
                other_point_id.resize(other_point_count);
                other_point_is_core.resize(other_point_count);
                MPI_Recv(&other_point_id[0], other_point_count, MPI_UNSIGNED_LONG, status.MPI_SOURCE, 6, MPI_COMM_WORLD, &status);
                MPI_Recv(&other_point_is_core[0], other_point_count, MPI_SHORT, status.MPI_SOURCE, 7, MPI_COMM_WORLD, &status);

                // TODO add into point_is_core_map
                addIntoMap(point_is_core_map, other_point_id, other_point_is_core);
            }
            // local_partition and local_adj_list for master now is the global graph
            std::vector<int> cell_cluster_id(local_partition.size(), 0);
            cluster_count = labelCoreCells(local_partition, local_adj_list, cell_cluster_id);
            labelPointsInCoreCells(cell_cluster_id, labels);
            // TODO label points in non core cells
            labelPointsInNonCoreCells(cell_cluster_id, point_is_core_map, labels);
        }

        MPI_Type_free(&MPI_Cell);
        MPI_Finalize();
        return cluster_count;
    }

private:
    HyperParameters params;
    std::vector<Vec2> points;
    size_t* point_index;
    size_t* cell_index;
    size_t* cell_start_index;
    size_t* cell_end_index;

    void setup(std::vector<Vec2> &points, float eps, size_t min_points) {
        params.eps = eps;
        params.min_points = min_points;
        params.num_points = points.size();

        // find lower left point and row_cells, col_cells and cell side length
        float min_x(0.), min_y(0.);
        float max_x(0.), max_y(0.);
        for(size_t i = 0; i < points.size(); ++i){
            Vec2 point = points[i];
            min_x = MIN(min_x, point.x);
            min_y = MIN(min_y, point.y);
            max_x = MAX(max_x, point.x);
            max_y = MAX(max_y, point.y);
        }
        params.min_x = min_x;
        params.min_y = min_y;
        params.side = eps * sqrt(2) / 2.f; // diagonal is eps
        params.row_cells = ceil((max_y - params.min_y)/params.side);
        params.col_cells = ceil((max_x - params.min_x)/params.side);
        params.num_cells = params.row_cells * params.col_cells;

        // allocate host data
        this -> points = points;
        this -> point_index = new size_t[params.num_points];
        this -> cell_index = new size_t[params.num_points];
        this -> cell_start_index = new size_t[params.num_cells];
        this -> cell_end_index = new size_t[params.num_cells];
        std::cout << "setup host success" << std::endl;

        // set up device data
        setup_device(&params, points);
    }

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

    void addIntoMap(std::unordered_map<size_t, short> point_is_core_map, std::vector<size_t>& point_id, std::vector<short>& point_is_core) {
        for(int i = 0; i < point_id.size(); i++) {
            point_is_core_map[point_id[i]] = point_is_core[i];
        }
    }

    // add a cell into the merged graph
    void addCell(Cell cell, std::vector<Cell>& all_cells, std::vector<size_t>& adj_list_sub, std::vector<size_t>& adj_list) {
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
    void mergeTwoGraph(std::vector<Cell>& partition1, std::vector<size_t>& adj_list1, 
                  std::vector<Cell>& partition2, std::vector<size_t>& adj_list2, 
                  std::vector<Cell>& all_cells, std::vector<size_t>& adj_list) {
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
    int labelCoreCells(std::vector<Cell>& all_cells, std::vector<size_t>& adj_list, std::vector<int>& cell_cluster_id) {
        int cluster_id = 0;
        std::vector<int> visited(all_cells.size(), 0);
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
             std::vector<int>& visited, 
             std::vector<Cell>& all_cells,
             std::vector<size_t>& adj_list, 
             int cluster_id,
             std::vector<int>& cell_cluster_id) {
        std::queue<int> queue;
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

    void labelPointsInCoreCells(std::vector<int>& cell_cluster_id, std::vector<int>& labels) {
        for(int cell_id = 0; cell_id < cell_cluster_id.size(); cell_id++) {
            if(cell_cluster_id[cell_id] > 0) {
                // label one core cell
                for(size_t i = cell_start_index[cell_id]; i < cell_end_index[cell_id]; i++) {
                    size_t point_id = point_index[i];
                    labels[point_id] = cell_cluster_id[cell_id];
                }
            }
        }
    }

    void labelPointsInNonCoreCells(std::vector<int>& cell_cluster_id, 
                                    std::unordered_map<size_t, short> point_is_core_map,
                                    std::vector<int>& labels) {
        for(int cell_id = 0; cell_id < cell_cluster_id.size(); cell_id++) {
            if(cell_cluster_id[cell_id] > 0) {
                // label one non core cell
                for(size_t i = cell_start_index[cell_id]; i < cell_end_index[cell_id]; i++) {
                    size_t point_id = point_index[i];
                    // label one point
                    // TODO
                    std::vector<size_t> neighbours = findNeighbours(point_id, cell_id);
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
    
    std::vector<size_t> findNeighbours(size_t point_id, int cell_id) {
        std::vector<size_t> neighbours;
        int cell_row_id = cell_id/params.col_cells;
        int cell_col_id = cell_id%params.col_cells;
        for(int row_diff = -2; row_diff < 3; ++row_diff) {
            for(int col_diff = -2; col_diff < 3; ++col_diff) {
                addOneCellNeighbours(neighbours, point_id, cell_row_id+row_diff, cell_col_id+col_diff);
            }
        }
    }

    void addOneCellNeighbours(std::vector<size_t>& neighbours, size_t point_id, int cell_row_id, int cell_col_id) {
        if(cell_row_id < 0 || cell_row_id > params.row_cells || cell_col_id < 0 || cell_col_id > params.col_cells) {
            return;
        }
        int cell_id = cell_row_id * params.col_cells + cell_col_id;
        for(size_t i = cell_start_index[cell_id]; i < cell_end_index[cell_id]; i++) {
            size_t other_point_id = point_index[i];
            if((points[other_point_id]-points[point_id]).length() <= params.eps) {
                neighbours.push_back(other_point_id);
            }
        }
    }

    std::vector<int>
    random_split(int num_partitions, int worker_id)
    {
        srand(RandSeed);
        std::vector<int> res;
        for(int cell_id = 0; cell_id < params.num_cells; ++cell_id){
            int partition_id = rand() % num_partitions;
            if(partition_id == worker_id){
                res.push_back(cell_id);
            }
        }
        return res;
    }
};

std::unique_ptr<DBScanner> createRPDBScanner(){
    return std::make_unique<RPDBScanner>();
}
