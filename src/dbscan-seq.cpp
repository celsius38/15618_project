#include <memory>
#include <deque>
#include "dbscan.h"

#include "make_unique.h"

class SequentialDBScanner: public DBScanner
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
        using std::deque;
        vector<vector<size_t>> allNeighbors = findNeighbors(points, eps);
        size_t counter = 0;  // current number of clusters
        for(size_t i = 0; i < points.size(); i++){
            auto point = points[i];
            auto label = labels[i];
            // already in a cluster, skip
            if(label > 0) continue;
            vector<size_t> &neighbors = allNeighbors[i];
            // noise
            if(neighbors.size() < minPts){
                labels[i] = -1;
                continue;
            }
            counter++;
            labels[i] = counter;
            deque<size_t> q(neighbors.begin(), neighbors.end());
            while(q.size() > 0){
                size_t j = q.front();
                q.pop_front();
                if(j == i) continue; // self, skip
                if(labels[j] > 0)continue; // other cluster, skip
                labels[j] = counter;
                vector<size_t> &subNeighbors = allNeighbors[j];
                if(subNeighbors.size() >= minPts){
                    q.insert(q.end(), subNeighbors.begin(), subNeighbors.end());
                }
            }
        }
        return counter;
    }

private:
    /* return the list of neighbors for each point,
     * a point's neighbors include the point itself
     */
    std::vector<std::vector<size_t>> findNeighbors(
        std::vector<Vec2> &points, float eps
    ){
        std::vector<std::vector<size_t>> neighbors(points.size());
        for(size_t i = 0; i < points.size(); i++){
            Vec2 p1 = points[i];
            std::vector<size_t> &p1_neighbors = neighbors[i];
            p1_neighbors.push_back(i);  // neighbors include the point itself
            for(size_t j = i+1; j < points.size(); j++){
                Vec2 p2 = points[j];
                std::vector<size_t> &p2_neighbors = neighbors[j];
                if((p1-p2).length() <= eps){
                    p1_neighbors.push_back(j);
                    p2_neighbors.push_back(i);
                }
            }
        }
        return neighbors;
    }
};

std::unique_ptr<DBScanner> createSequentialDBScanner(){
    return std::make_unique<SequentialDBScanner>();
}


