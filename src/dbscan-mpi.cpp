#include <memory>
#include "dbscan.h"
#include "mpi.h"
#include "make_unique.h"

class RPDBScanner: public DBScanner
{
public: 
    /* Return total number of clusters
     * insert corresponding cluster id in `labels`
     * -1 stands for noise, 0 for unprocessed, otherwise stands for the cluster id
     */
    size_t scan(
        std::vector<Vec2> &points, std::vector<int> &labels, float eps, size_t minPts
    ){ 
        // data partition
        // build local clustering
        // merge clustering
    }

private:
    




};

std::unique_ptr<DBScanner> createRPDBScanner(){
    return std::make_unique<RPDBScanner>();
}
