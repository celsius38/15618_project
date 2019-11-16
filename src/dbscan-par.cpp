#include <memory>
#include "dbscan.h"

#include "make_unique.h"

void bfs_cuda(size_t* vertex_degree, 
                size_t* vertex_start_index,
                size_t* adj_list, 
                size_t* boarder,  
                size_t minPts, 
                int* labels, 
                int counter,
                size_t N,
                size_t adj_list_len);

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
            costruct_graph(vertex_degree, vertex_start_index, points, eps);

        size_t counter = 0;  // current number of clusters
        for(size_t i = 0; i < points.size(); i++){
            auto label = labels[i];
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

    bool isEmpty(std::vector<size_t> boarder) {
        for(size_t i = 0; i < boarder.size(); i++) {
            if(boarder[i]) {
                return false;
            }
        }
        return true;
    }

    void bfs(size_t i, 
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
            bfs_cuda(vertex_degree.data(), 
                vertex_start_index.data(), 
                adj_list.data(), 
                boarder.data(), 
                minPts, 
                labels.data(), 
                counter,
                vertex_degree.size(),
                adj_list.size());
        }
    }

    void degree_kernel(size_t v, std::vector<size_t> &vertex_degree, std::vector<Vec2> &points, float eps) {
        size_t degree = 0;
        Vec2 p1 = points[v];
        for(size_t i = 0; i < points.size(); i++){
            Vec2 p2 = points[i];
            if((p1-p2).length() <= eps){
                degree++;
            }
        }
        vertex_degree[v] = degree;
    }

    void adj_list_kernel(size_t v, 
                        std::vector<size_t> &vertex_start_index, 
                        std::vector<size_t> &adj_list,
                        std::vector<Vec2> &points, 
                        float eps) {
        size_t cur_index = vertex_start_index[v];
        Vec2 p1 = points[v];
        for(size_t i = 0; i < points.size(); i++) {
            Vec2 p2 = points[i];
            if((p1-p2).length() <= eps){
                adj_list[cur_index] = i;
                cur_index++;
            }
        }
    }

    std::vector<size_t> costruct_graph(std::vector<size_t> &vertex_degree, 
                                        std::vector<size_t> &vertex_start_index, 
                                        std::vector<Vec2> &points, 
                                        float eps) {
        for(size_t v = 0; v < points.size(); v++) {
            // TODO: CUDA version removes v
            degree_kernel(v, vertex_degree, points, eps);
        }
        // TODO: exclusive scan
        size_t cur_index = 0;
        for(size_t v = 0; v < points.size(); v++) {
            vertex_start_index[v] = cur_index;
            cur_index += vertex_degree[v];
        }
        std::vector<size_t> adj_list(cur_index);
        for(size_t v = 0; v < points.size(); v++) {
            // TODO: CUDA version removes v
            adj_list_kernel(v, vertex_start_index, adj_list, points, eps);
        }
        return adj_list;
    }
};

std::unique_ptr<DBScanner> createParallelDBScanner(){
    return std::make_unique<ParallelDBScanner>();
}
