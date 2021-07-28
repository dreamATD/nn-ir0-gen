#include <iostream>
#include "models.hpp"
#include "neuralNetwork.hpp"

#define IN          1
#define INS_FILE    2
#define WIT_FILE    3
#define REL_FILE    4
#define MODE        5

int QSIZE = 51;
int SCALE = 10;
string mode; // "nchw" or "hwcn"

int main(int argc, char **argv) {
    mode = argv[MODE];
//    vgg16 model(32, 32, 3, 1, NAIVE, MAX, argv[IN]);
    lenet model(28, 28, 1, 1, NAIVE, MAX, argv[IN]);
    layeredCircuit C;
    model.create(C);

    C.print(argv[INS_FILE], argv[WIT_FILE], argv[REL_FILE]);
    return 0;
}
