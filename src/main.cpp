#include <iostream>
#include <cstring>
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

vector<int> vgg11_config{1, 1, 2, 2, 2};
vector<int> vgg13_config{2, 2, 2, 2, 2};
vector<int> vgg16_config{2, 2, 3, 3, 3};
vector<int> vgg19_config{2, 2, 4, 4, 4};

int main(int argc, char **argv) {
    mode = argv[MODE];

    circuit C;
    if (strcmp(argv[IN], "vgg1") == 0) {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], {1, 0, 0, 0, 0});
        model.create(C);
    } else if (strcmp(argv[IN], "vgg2") == 0) {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], {1, 1, 0, 0, 0});
        model.create(C);
    } else if (strcmp(argv[IN], "vgg3") == 0) {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], {1, 1, 1, 0, 0});
        model.create(C);
    } else if (strcmp(argv[IN], "vgg4") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], {1, 1, 1, 1, 0});
        model.create(C);
    } else if (strcmp(argv[IN], "vgg5") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], {1, 1, 1, 1, 1});
        model.create(C);
    } else if (strcmp(argv[IN], "vgg11") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], vgg11_config);
        model.create(C);
    } else if (strcmp(argv[IN], "vgg13") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], vgg13_config);
        model.create(C);
    } else if (strcmp(argv[IN], "vgg16") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], vgg16_config);
        model.create(C);
    } else if (strcmp(argv[IN], "vgg19") == 0)  {
        vgg model(32, 32, 3, 1, SQR_ACT, NAIVE, SUM, argv[IN], vgg19_config);
        model.create(C);
    } else if (strcmp(argv[IN], "lenet") == 0) {
        lenet model(28, 28, 1, 1, SQR_ACT, NAIVE, SUM, argv[IN]);
        model.create(C);
    }

    C.print(argv[INS_FILE], argv[WIT_FILE], argv[REL_FILE]);
    return 0;
}
