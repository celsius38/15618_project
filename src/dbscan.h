#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <memory>
#include "utils.h"

class DBScanner{
public:
    //pure virtual function
    virtual size_t scan(std::vector<Vec2> &points, std::vector<int> &labels, float eps,  size_t minPts) = 0;
    virtual ~DBScanner(){};
};

std::unique_ptr<DBScanner> createSequentialDBScanner();
std::unique_ptr<DBScanner> createGDBScanner();
std::unique_ptr<DBScanner> createRPDBScanner();

struct HyperParameters {
    float eps;
    size_t min_points;
    size_t num_points;

    float min_x;
    float min_y;
    float side;
    int num_cells;
    int row_cells;
    int col_cells;

    float* points; 
    size_t* point_index;
    size_t* cell_index;
    size_t* cell_start_index;
    size_t* cell_end_index;
};

#endif
