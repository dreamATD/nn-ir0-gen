//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_VGG_HPP
#define ZKCNN_VGG_HPP

#include "neuralNetwork.hpp"

class vgg: public neuralNetwork {
public:
    explicit vgg(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
                   const std::string &i_filename, const vector<int> &nn_config);

};

class lenet: public neuralNetwork {
public:
    explicit lenet(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
                   const std::string &i_filename);
};

class ccnn: public neuralNetwork {
public:
    explicit ccnn(int psize_x, int psize_y, int pparallel, int pchannel, poolType pool_ty,
                  const std::string &filename);
};

#endif //ZKCNN_VGG_HPP
