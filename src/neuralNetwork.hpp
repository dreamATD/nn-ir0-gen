//
// Created by 69029 on 3/16/2021.
//

#ifndef ZKCNN_NEURALNETWORK_HPP
#define ZKCNN_NEURALNETWORK_HPP

#include <vector>
#include <fstream>
#include <string>
#include "circuit.h"
#include "typedef.hpp"

using std::vector;
using std::string;
using std::tuple;
using std::pair;

enum convType {
    FFT, NAIVE, NAIVE_FAST
};

struct convKernel {
    convType ty;
    int channel_out, channel_in, size, stride_bl, padding;
    convKernel(convType _ty, int _channel_out, int _channel_in, int _size, int _log_stride, int _padding) :
            ty(_ty), channel_out(_channel_out), channel_in(_channel_in), size(_size), stride_bl(_log_stride), padding(_padding){}

    convKernel(convType _ty, int _channel_out, int _channel_in, int _size):
            convKernel(_ty, _channel_out, _channel_in, _size, 0, _size >> 1) {}
};

struct fconKernel {
    int channel_out, channel_in;
    fconKernel(int _channel_out, int _channel_in):
        channel_out(_channel_out), channel_in(_channel_in) {}
};

enum poolType {
    AVG, MAX, NONE
};

enum actType {
    RELU_ACT
};

struct poolKernel {
    poolType ty;
    int size, stride_bl;
    poolKernel(poolType _ty, int _size, int _log_stride):
            ty(_ty), size(_size), stride_bl(_log_stride) {}
};


class neuralNetwork {
public:
    explicit neuralNetwork(int psize_x, int psize_y, int pchannel, int pparallel, const string &i_filename);
    neuralNetwork(int psize, int pchannel, int pparallel, int kernel_size, int sec_size, int fc_size,
                  int start_channel, convType conv_ty, poolType pool_ty);
    void create(circuit &C);

protected:

    void refreshConvParam(i64 new_nx, i64 new_ny, i64 new_m, i64 new_chan_in, i64 new_chan_out,
                          i64 new_log_stride, i64 new_padding);
    void refreshConvParam(i64 new_nx, i64 new_ny, const convKernel &conv);
    void calcSizeAfterPool(const poolKernel &p);
    void refreshFCParam(i64 new_chan_in, i64 new_chan_out);
    void refreshFCParam(const fconKernel &fc);

    [[nodiscard]] i64 getPoolDecmpSize() const;
    void prepareBit(circuit &C, i64 data, i64 &dcmp_id, i64 bit_shift);
    void prepareSignBit(circuit &C, i64 data, i64 &dcmp_id);
    void prepareMax(circuit &C, i64 data, i64 &max_id);
    i64 updateGate(circuit &C, GateType ty, i64 u, i64 v, bool is_assert = false);

    vector<vector<convKernel>> conv_section;
    vector<poolKernel> pool;
    poolType pool_ty;
    i64 pool_bl, pool_sz;
    i64 pool_stride_bl, pool_stride;

    vector<fconKernel> full_conn;

    int pic_size_x, pic_size_y, pic_channel, pic_parallel;
    int Q_MAX, T;
    const int Q_BIT_SIZE = 220;

    int nx_in, nx_out, ny_in, ny_out, m, channel_in, channel_out, log_stride, padding;
    int new_nx_in, new_ny_in;
    int nx_padded_in, ny_padded_in;

    vector<i64> val;
    vector<i64>::iterator two_mul;

    vector<vector<vector<i64>>> inputLayer(circuit &C);
    vector<vector<vector<i64>>> naiveConvLayer(circuit &C, const vector<vector<vector<i64>>> &data);
    vector<vector<vector<i64>>> reluActConvLayer(circuit &C, const vector<vector<vector<i64>>> &data);
    vector<i64> reluActFconLayer(circuit &C, const vector<i64> &data);
    vector<vector<vector<i64>>> avgPoolingLayer(circuit &C, const vector<vector<vector<i64>>> &data);
    vector<vector<vector<i64>>> maxPoolingLayer(circuit &C, const vector<vector<vector<i64>>> &data);
    vector<i64> fullyConnLayer(circuit &C, const vector<i64> &data);
    i64 multiOpt(circuit &C, GateType ty, const vector<i64> &list, bool is_assert = false);
    vector<i64> equalCheck(circuit &C, const vector<i64> &data, const vector<vector<i64>> &bits, bool has_sign, bool need_neg);
    vector<vector<i64>> bitCheck(circuit &C, const vector<vector<i64>> &bits);
    vector<i64> rescaleData(circuit &C, const vector<vector<i64>> &bits, bool has_sign);
    vector<vector<i64>> bitDecomposition(circuit &C, const vector<i64> &data, int nbits, bool has_sign);
    vector<vector<i64>> bitValueDecomposition(circuit &C, const vector<i64> &data, int nbits, bool has_sign);
    vector<vector<i64>> scaledBits(circuit &C, const vector<vector<i64>> &bits, bool neg);
};


#endif //ZKCNN_NEURALNETWORK_HPP
