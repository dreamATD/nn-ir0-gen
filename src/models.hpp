//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_VGG_HPP
#define ZKCNN_VGG_HPP

#include "neuralNetwork.hpp"

class vgg16: public neuralNetwork {

public:
    explicit vgg16(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, convType conv_ty, poolType pool_ty,
                   const std::string &i_filename);

};

class vgg11: public neuralNetwork {

public:
    explicit vgg11(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, convType conv_ty, poolType pool_ty,
                   const std::string &i_filename);

};

class lenet: public neuralNetwork {
public:
    explicit lenet(i64 psize_x, i64 psize_y, i64 pchannel, i64 pparallel, convType conv_ty, poolType pool_ty,
                   const std::string &i_filename);
};

class ccnn: public neuralNetwork {
public:
    explicit ccnn(i64 psize_x, i64 psize_y, i64 pparallel, i64 pchannel, poolType pool_ty,
                  const std::string &filename);
};

#endif //ZKCNN_VGG_HPP
