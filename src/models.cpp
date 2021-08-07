//
// Created by 69029 on 3/16/2021.
//

#include <tuple>
#include <iostream>
#include <cassert>
#include "models.hpp"
#include "utils.hpp"
#undef USE_VIRGO

vgg11::vgg11(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
             const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    int start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;

    // channel = 64 (start)
    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 128 (start << 1)
    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 256 (start << 2)
    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

vgg13::vgg13(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
             const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    int start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;

    // channel = 64 (start)
    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    conv_section[0].emplace_back(conv_ty, start,  start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 128 (start << 1)
    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    conv_section[1].emplace_back(conv_ty, start << 1,  start << 1, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 256 (start << 2)
    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

vgg16::vgg16(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
             const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    int start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;

    // channel = 64 (start)
    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    conv_section[0].emplace_back(conv_ty, start, start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 128 (start << 1)
    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    conv_section[1].emplace_back(conv_ty, start << 1, start << 1, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 256 (start << 2)
    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

vgg19::vgg19(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
             const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename) {
    assert(psize_x == psize_y);
    conv_section.resize(5);

    int start = 64, kernel_size = 3, new_nx = pic_size_x, new_ny = pic_size_y;

    // channel = 64 (start)
    conv_section[0].emplace_back(conv_ty, start,  pic_channel, kernel_size);
    conv_section[0].emplace_back(conv_ty, start, start, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 128 (start << 1)
    conv_section[1].emplace_back(conv_ty, start << 1,  start, kernel_size);
    conv_section[1].emplace_back(conv_ty, start << 1, start << 1, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 256 (start << 2)
    conv_section[2].emplace_back(conv_ty, start << 2, start << 1, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    conv_section[2].emplace_back(conv_ty, start << 2, start << 2, kernel_size);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[3].emplace_back(conv_ty, start << 3, start << 2, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[3].emplace_back(conv_ty, start << 3, start << 3, 3);
    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    // channel = 512 (start << 3)
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);
    conv_section[4].emplace_back(conv_ty, start << 3, start << 3, 3);

    pool.emplace_back(pool_ty, 2, 1);
    new_nx = ((new_nx - pool.back().size) >> pool.back().stride_bl) + 1;
    new_ny = ((new_ny - pool.back().size) >> pool.back().stride_bl) + 1;

    if (pic_size_x == 224) {
        full_conn.emplace_back(4096, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(4096, 4096);
        full_conn.emplace_back(1000, 4096);
    } else {
        assert(pic_size_x == 32);
        full_conn.emplace_back(512, new_nx * new_ny * (start << 3));
        full_conn.emplace_back(512, 512);
        full_conn.emplace_back(10, 512);
    }
}

ccnn::ccnn(int psize_x, int psize_y, int pparallel, int pchannel, poolType pool_ty,
           const std::string &filename) : neuralNetwork(psize_x, psize_y, pchannel, pparallel, filename) {
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

lenet::lenet(int psize_x, int psize_y, int pchannel, int pparallel, convType conv_ty, poolType pool_ty,
             const std::string &i_filename)
        : neuralNetwork(psize_x, psize_y, pchannel, pparallel, i_filename) {

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