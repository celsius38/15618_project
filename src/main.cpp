#include <string.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>

#include "CycleTimer.h"
#include "utils.h"
#include "dbscan.h"

enum class ScannerType{
    Sequential,
    Seq2,
    Parallel
};

std::ostream& operator<<(std::ostream& out, const ScannerType &value){
    const char* s = 0;
#define PROCESS_VAL(p) case(p): s = #p; break;
    switch(value){
        PROCESS_VAL(ScannerType::Parallel);

    }
#undef PROCESS_VAL
    return out << s;
}

std::string plainScannerType(ScannerType type) {
    switch(type){
        case ScannerType::Sequential:
            return "sequential";
        case ScannerType::Seq2: 
            return "seq2";
        case ScannerType::Parallel:
            return "parallel";
    }
}

struct Options
{ 
    std::string inFile;
    float eps = 1;
    int minPts = 10;
    ScannerType scannerType = ScannerType::Sequential;
};

std::string removeQuote(std::string input)
{
    if (input.length() > 0 && input.front() == '\"')
        return input.substr(1, input.length() - 2);
    return input;
}

Options parseOptions(int argc, const char ** argv){
    Options opt;
    for(int i = 1; i < argc; i++){
        if (strcmp(argv[i],  "--input") == 0){
            opt.inFile = removeQuote(argv[i+1]);
        }else if(strcmp(argv[i], "--eps") == 0){
            opt.eps = (float)atof(argv[i+1]);
        }else if(strcmp(argv[i], "--minPts") == 0){
            opt.minPts = (int)atoi(argv[i+1]);
        }else if(strcmp(argv[i], "--seq") == 0){
            opt.scannerType = ScannerType::Sequential;
        }else if(strcmp(argv[i], "--seq2") == 0) {
            opt.scannerType = ScannerType::Seq2;
        }else if(strcmp(argv[i], "--par") == 0) {
            opt.scannerType = ScannerType::Parallel;
        }
    }
    if(opt.inFile.empty()){
        std::cerr << "Please specify input file --input" << std::endl;
        exit(EXIT_FAILURE);
    }
    return opt;
}

std::vector<Vec2> loadFromFile(std::string fileName){
    std::ifstream inFile;
    inFile.open(fileName);
    if(!inFile){
        std::cerr << "File not found" << std::endl;
        exit(EXIT_FAILURE);
    }
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

/*
 * given a file path, write the corresponding output points
 */
int main(int argc, const char ** argv){
    Options options = parseOptions(argc, argv);
    std::cout << "inFile: " << options.inFile << std::endl;
    std::cout << "scannerType: " << plainScannerType(options.scannerType) << std::endl;
    std::cout << "eps: " << options.eps << std::endl;
    std::cout << "minPts: " << options.minPts << std::endl;
    // load points
    std::vector<Vec2> points = loadFromFile(options.inFile);
    // choose scanner type
    std::unique_ptr<DBScanner> scanner;
    switch(options.scannerType){
        case ScannerType::Sequential:
            scanner = createSequentialDBScanner();
            break;
        case ScannerType::Seq2:
            scanner = createSeq2DBScanner();
            break;
        case ScannerType::Parallel:
            scanner = createParallelDBScanner();
            break;
    }
    // scan
    std::vector<int> labels(points.size(), 0);
    double start = CycleTimer::currentMSeconds(); 
    size_t numClusters = scanner->scan(points, labels, options.eps, options.minPts);
    std::cout << "Taking " << CycleTimer::currentMSeconds()-start << " ms"<< std::endl;
    std::cout << "====================" << std::endl;
    std::cout << numClusters << " clusters" << std::endl; 
    for(auto label: labels){
        std::cout << label << std::endl;
    }
    return 0;
}

