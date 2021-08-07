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
string mode; // "nchw" for vgg16 or "hwcn" for lenet

int main(int argc, char **argv) {
    mode = argv[MODE];
    string i_filename = argv[IN];

    circuit C;
    if (i_filename.find("vgg11") != string::npos) {
        vgg16 model(32, 32, 3, 1, NAIVE, MAX, argv[IN]);
        model.create(C);
    } else if (i_filename.find("vgg13") != string::npos) {
        vgg16 model(32, 32, 3, 1, NAIVE, MAX, argv[IN]);
        model.create(C);
    } else if (i_filename.find("vgg16") != string::npos) {
        vgg16 model(32, 32, 3, 1, NAIVE, MAX, argv[IN]);
        model.create(C);
    } else if (i_filename.find("vgg19") != string::npos) {
        vgg16 model(32, 32, 3, 1, NAIVE, MAX, argv[IN]);
        model.create(C);
    } else if (i_filename.find("lenet") != string::npos) {
        lenet model(28, 28, 1, 1, NAIVE, MAX, argv[IN]);
        model.create(C);
    }

    C.print(argv[INS_FILE], argv[WIT_FILE], argv[REL_FILE]);
    return 0;
}
