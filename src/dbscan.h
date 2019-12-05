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

#endif
