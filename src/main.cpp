#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "utils.h"
#include "dbscan.h"

enum class ScannerType{
    Sequential
};

struct Options
{ 
    std::string inFile;
    std::string outFile;
    ScannerType scannerType = ScannerType::Sequential;
}

Options parseOptions(int argc, const char ** argv){
}

/*
 * given a file path, write the corresponding output points
 */
int main(int argc, const char ** argv){
    Options options = parseOptions(argc, argv);

    std::string input(argv[0]);
    std::vector<Vec2> points = loadFromFile(input);
    if(!points) 
        throw "File not found";
    
}

std::vector<Vec2> loadFromFile(std::string fileName){
    std::ifstream inFile;
    inFile.open(fileName);
    if(!inFile) return nullptr;
    
    std::string line;
    std::vector<Vec2> points;
    while(std::getline(inFile, line)){
        std::stringstream sstream(line);
        std::string str;
        Vec2 point;
        std::getline(sstream, str, ' ');
        point.x = (float)atof(str.c_str());
        std::getline(sstream, str, ' ');
        point.y = (float)atof(str.c_str()); 
        points.push_back(point);
    }
    inFile.close();
    return points;
}

