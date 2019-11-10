#ifndef DBSCAN_H
#define DBSCAN_H

#include "world.h"

class DBScanner{
public:
    //pure virtual function
    virtual void scan(std::vector<Vec2> &points, std::vector<Vec2> &label, float eps,  int minPts) = 0;
    virtual ~DBScanner(){}
}

std::unique_ptr<INBodySimulator> createSequentialDBScanner();
#endif
