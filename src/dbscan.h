#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <memory>
#include "dbscan.h"
#include "utils.h"

class DBScanner{
public:
    //pure virtual function
    virtual size_t scan(std::vector<Vec2> &points, std::vector<int> &labels, float eps,  int minPts) = 0;
    virtual ~DBScanner(){};
};

std::unique_ptr<DBScanner> createSequentialDBScanner();


#endif
