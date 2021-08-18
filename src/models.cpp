//
// Created by 69029 on 3/16/2021.
//

#include <tuple>
#include <iostream>
#include <cassert>
#include "models.hpp"
#include "utils.hpp"

vgg::vgg(int psize_x, int psize_y, int pchannel, int pparallel, actType act_ty, convType conv_ty, poolType pool_ty,
         const std::string &i_filename, const vector<int> &nn_config)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, act_ty, i_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    int previous = pic_channel, start = 16, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;

    // channel = 64 (start)
    for (int k = 0; k < 5; ++k) {
        if (nn_config[k]) {
            conv_section[k].emplace_back(conv_ty, start,  previous, kernel_size);
            for (int i = 1; i < nn_config[k]; ++i)
                conv_section[k].emplace_back(conv_ty, start,  start, kernel_size);
            pool.emplace_back(pool_ty, 2, 1);
            previous = start;
            if (k < 3) start <<= 1;
        }
        new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
        new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;
    }

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * previous);
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * previous);
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

ccnn::ccnn(int psize_x, int psize_y, int pparallel, int pchannel, poolType pool_ty,
           const std::string &filename) : neuralNetwork(psize_x, psize_y, pchannel, pparallel, act_ty, filename) {
    conv_section.resize(1);

    conv_section[0].emplace_back(NAIVE_FAST, 2,  pchannel, 3, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

//    conv_section[1].emplace_back(FFT, 64, 4, 3);
//    conv_section[1].emplace_back(NAIVE, 64,  64, 3);
//    pool.emplace_back(pool_ty, 2, 1);

//    conv_section[0].emplace_back(FFT, 2, pic_channel, 3);
//    conv_section[1].emplace_back(NAIVE, 1,  2, 3);
//    pool.emplace_back(pool_ty, 2, 1);
}

lenet::lenet(int psize_x, int psize_y, int pchannel, int pparallel, actType act_ty, convType conv_ty,
             poolType pool_ty, const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, act_ty, i_filename) {

    conv_section.emplace_back();
    conv_section[0].emplace_back(conv_ty, 6,  pchannel, 5, 0, 2);
    pool.emplace_back(pool_ty, 2, 1);

    conv_section.emplace_back();
    conv_section[1].emplace_back(conv_ty, 16,  6, 5, 0, 0);
    pool.emplace_back(pool_ty, 2, 1);

//    conv_section.emplace_back();
//    conv_section[2].emplace_back(conv_ty, 120,  16, 5, 0, 0);
//    pool.emplace_back(pool_ty, 2, 1);

    full_conn.emplace_back(120, 400);
    full_conn.emplace_back(84, 120);
    full_conn.emplace_back(10, 84);
}