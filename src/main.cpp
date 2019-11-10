#include <sstream>
#include <iostream>
#include "world.h"

int main(int argc, const char ** argv){
    std::cout << "This is the main" << std::endl;
    Vec2 u(3,4);
    std::cout << u.length() << std::endl;
}
