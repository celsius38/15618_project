#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include "dbscan.h"
#include "world.h"

class DBScanner{
public:
    //pure virtual function
    virtual void scan(std::vector<Vec2> &points, std::vector<int> &labels, float eps,  int minPts) = 0;
    virtual ~DBScanner(){};
};

std::unique_ptr<DBScanner> createSequentialDBScanner();

#endif
