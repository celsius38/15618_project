#include <memory>
#include "dbscan.h"

class SequentialDBScanner: public DBScanner
{
public: 
    void scan(
        std::vector<Vec2> &points, std::int<Vec2> &labels, float eps, int minPts
    ){
        // -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
        int counter = 0;  // current number of clusters
        for(size_t i = 0; i < points.size(); i++){
            auto point = points[i];
            auto label = labels[i];
            // already in a cluster, skip
            if(label > 0) continue;
            // find number of neighbors
            
        }
    }
};
