//
// Created by 69029 on 3/16/2021.
//

#include "neuralNetwork.hpp"
#include "utils.hpp"
#include "global_var.hpp"
#include <iostream>
#include <cmath>

using std::cerr;
using std::endl;
using std::max;
using std::ifstream;
using std::ofstream;

ifstream in;
neuralNetwork::neuralNetwork(int psize_x, int psize_y, int pchannel, int pparallel, const string &i_filename) :
        pic_size_x(psize_x), pic_size_y(psize_y), pic_channel(pchannel), pic_parallel(pparallel),
        T(SCALE), Q_MAX(SCALE + QSIZE) {
    in.open(i_filename);
    if (!in.is_open())
        fprintf(stderr, "Can't find the input file. \n");
}

neuralNetwork::neuralNetwork(int psize, int pchannel, int pparallel, int kernel_size, int sec_size, int fc_size,
                             int start_channel, convType conv_ty, poolType pool_ty)
        : neuralNetwork(psize, psize, pchannel, pparallel, "") {
    pool_bl = 2;
    pool_stride_bl = pool_bl >> 1;
    conv_section.resize(sec_size);

    i64 start = start_channel;
    for (i64 i = 0; i < sec_size; ++i) {
        conv_section[i].emplace_back(conv_ty, start << i, i ? (start << (i - 1)) : pic_channel, kernel_size);
        conv_section[i].emplace_back(conv_ty, start << i, start << i, kernel_size);
        pool.emplace_back(pool_ty, 2, 1);
    }

    i64 new_nx = (pic_size_x >> pool_stride_bl * conv_section.size());
    i64 new_ny = (pic_size_y >> pool_stride_bl * conv_section.size());
    for (i64 i = 0; i < fc_size; ++i)
        full_conn.emplace_back(i == fc_size - 1 ? 1000 : 4096, i ? 4096 : new_nx * new_ny * (start << (sec_size - 1)));
}

vector<i64> flatten(const vector<vector<vector<i64>>> &data, const string &special_mode) {
    vector<i64> res;
    if (special_mode == "hwcn") {
        for (auto &x_vec: data)
            for (auto &y_vec: x_vec)
                for (auto co: y_vec)
                    res.push_back(co);
    } else {
        for (auto &co_vec: data)
            for (auto &x_vec: co_vec)
                for (auto y: x_vec)
                    res.push_back(y);
    }
    return res;
}

void neuralNetwork::create(circuit &C) {
    // assert(pool.size() >= conv_section.size() - 1);

    C.init(Q_BIT_SIZE);
    two_mul = C.two_mul.begin();
    auto data = inputLayer(C);

    new_nx_in = pic_size_x;
    new_ny_in = pic_size_y;

    for (i64 i = 0; i < conv_section.size(); ++i) {
        auto &sec = conv_section[i];
        if (sec.size() == 0) continue;
        for (i64 j = 0; j < sec.size(); ++j) {
            auto &conv = sec[j];
            refreshConvParam(new_nx_in, new_ny_in, conv);
            pool_ty = i < pool.size() && j == sec.size() - 1 ? pool[i].ty : NONE;
            data = naiveConvLayer(C, data);
            if (j != sec.size() - 1)
                data = reluActConvLayer(C, data);
        }
        calcSizeAfterPool(pool[i]);
        data = maxPoolingLayer(C, data);
    }

    // vector<i64> data_flatten(flatten(data, mode));
    // pool_ty = NONE;
    // for (int i = 0; i < full_conn.size(); ++i) {
    //     auto &fc = full_conn[i];
    //     refreshFCParam(fc);
    //     data_flatten = fullyConnLayer(C, data_flatten);
    //     if (i == full_conn.size() - 1) break;
    //     data_flatten = reluActFconLayer(C, data_flatten);
    // }

    cerr << "finish creating circuit." << endl;
}

i64 neuralNetwork::updateGate(circuit &C, GateType ty, i64 u, i64 v, bool is_assert) {
    i64 res;
    double num;
    switch (ty) {
        case Ins:
            if (!(in >> num)) num = 1;
            res = i64(num * exp2(SCALE));
            u = res;
            break;
        case Wit:
            res = u;
            break;
        case Add:
            res = (val[u] + val[v]);
            break;
        case Mul:
            res = (i128) val[u] * val[v];
            break;
        case Addc:
            res = (val[u] + v);
            break;
        case Mulc:
            res = val[u] * v;
            break;
        default:
            assert(false);
    }
    C.gates.emplace_back(ty, u, v, is_assert);
    val.push_back(res);
    return C.gates.size() - 1;
}

i64 neuralNetwork::multiOpt(circuit &C, GateType ty, const vector<i64> &list, bool is_assert) {
    auto vec = list;
    while (vec.size() > 1) {
        for (int i = 0; i < vec.size(); ++i) {
            for (int j = 0; j + 1 < vec.size(); j += 2) {
                vec[j >> 1] = updateGate(C, ty, vec[j], vec[j + 1]);
            }
            if (vec.size() & 1) vec[(vec.size() >> 1)] = vec.back();
            i64 new_size = (vec.size() + 1) >> 1;
            vec.resize(new_size);
        }
    }
    C.gates[vec[0]].is_assert = is_assert;
    return vec[0];
}

vector<i64> neuralNetwork::equalCheck(circuit &C, const vector<i64> &data, const vector<vector<i64>> &bits, bool has_sign, bool need_neg) {
    vector<i64> data_p;
    if (has_sign) {
        // equal check: 2 * sign_bit
        vector<i64> dsign(data.size());
        for (i64 i = 0; i < data.size(); ++i) {
            dsign[i] = updateGate(C, Mulc, bits[i][0], 2);
        }

        // equal check: 2 * sign_bit - 1
        vector<i64> dsign_1(data.size());
        for (i64 i = 0; i < data.size(); ++i) {
            dsign_1[i] = updateGate(C, Addc, dsign[i], -1);
        }

        // equal check: (2 * sign_bit - 1) * data
        vector<i64> dsign_1xdata(data.size());
        for (i64 i = 0; i < data.size(); ++i) {
            dsign_1xdata[i] = updateGate(C, Mul, dsign_1[i], data[i]);
        }
        data_p = dsign_1xdata;
    } else if (need_neg) {
        data_p.resize(data.size());
        for (int i = 0; i < data.size(); ++i)
            data_p[i] = updateGate(C, Mulc, data[i], -1);
    } else data_p = data;

    // equal check: 2^k * data_bit_k
    auto cbits = scaledBits(C, bits, need_neg);

    // equal check: (2 * sign_bit - 1) * data + 2^0 * data_bit_0 + ... + 2^{Q_MAX-2} * data_bit_{Q_MAX-2}
    vector<i64> equal_check(data.size());
    for (i64 i = 0; i < data.size(); ++i) {
        equal_check[i] = multiOpt(C, Add, cbits[i], true);
    }
    return equal_check;
}

vector<vector<i64>> neuralNetwork::bitCheck(circuit &C, const vector<vector<i64>> &bits) {
    // bit check: bit - 1
    i64 size = bits.size();
    vector<vector<i64>> bits_1(size);
    for (i64 i = 0; i < size; ++i) {
        bits_1[i].resize(Q_MAX);
        for (int j = 0; j < Q_MAX; ++j) {
            bits_1[i][j] = updateGate(C, Addc, bits[i][j], -1);
        }
    }

    // bit check: (bit - 1) * bit
    vector<vector<i64>> bit_check(size);
    for (i64 i = 0; i < size; ++i) {
        bit_check[i].resize(Q_MAX);
        for (int j = 0; j < Q_MAX; ++j) {
            bit_check[i][j] = updateGate(C, Mul, bits_1[i][j], bits[i][j], true);
        }
    }
    return bit_check;
}

vector<i64> neuralNetwork::rescaleData(circuit &C, const vector<vector<i64>> &bits, bool has_sign) {
    // rescale: -sign_bit
    i64 size = bits.size();
    vector<i64> neg_sign_bit(size);
    for (i64 i = 0; i < size; ++i) {
        neg_sign_bit[i] = updateGate(C, Mulc, bits[i][0], -1);
    }

    // rescale: (-sign_bit + 1)
    vector<i64> neg_sign_bit_1(size);
    for (i64 i = 0; i < size; ++i) {
        neg_sign_bit_1[i] = updateGate(C, Addc, neg_sign_bit[i], 1);
    }

    // rescale: 2^0 * data_bit_T + ... + 2^{Q_MAX-2-T} * data_bit_{Q_MAX-2}
    vector<i64> rescale(size);
    for (i64 i = 0; i < size; ++i) {
        vector<i64> tmp(Q_MAX - 1 - T);
        for (int j = 1; j < Q_MAX - T; ++j) {
            tmp[j - 1] = updateGate(C, Mulc, bits[i][j], two_mul[Q_MAX - j - T - 1]);
//            fprintf(stderr, "%lld %lld %lld %d\n", val[tmp[j]], val[bits[i][j]], two_mul[Q_MAX - j - T - 1], Q_MAX - j - T - 1);
        }
        rescale[i] = multiOpt(C, Add, tmp);
    }

    if (!has_sign) {
//        fprintf(stderr, "rescale: \n");
//        for (auto x: rescale) {
//            fprintf(stderr, "%lld ", val[x]);
//        }
//        fprintf(stderr, "\n");
        return rescale;
    }
    vector<i64> sign_rescale(size);

    // rescale: (-sign_bit + 1) (2^0 * data_bit_T + ... + 2^{Q_MAX-2-T} * data_bit_{Q_MAX-2})
    for (i64 i = 0; i < size; ++i)
        sign_rescale[i] = updateGate(C, Mul, neg_sign_bit_1[i], rescale[i]);

//    fprintf(stderr, "rescale: \n");
//    for (auto x: sign_rescale) {
//        fprintf(stderr, "%lld ", val[x]);
//    }
//    fprintf(stderr, "\n");
    return sign_rescale;
}

vector<vector<i64>> neuralNetwork::bitDecomposition(circuit &C, const vector<i64> &data, int nbits, bool has_sign) {
    // bit decomposition
    if (!has_sign) nbits += 1;
    vector<vector<i64>> bits(data.size());
    for (i64 i = 0; i < data.size(); ++i) {
        if (has_sign) {
            bits[i].resize(nbits);
            prepareSignBit(C, val[data[i]], bits[i][0]);
        } else bits[i].resize(nbits);
        for (int j = 1; j < nbits; ++j) {
            prepareBit(C, val[data[i]], bits[i][j], nbits - 1 - j);
        }
    }
//    fprintf(stderr, "Bits:\n");
//    for (i64 i = 0; i < data.size(); ++i) {
//        fprintf(stderr, "(%lld): ", val[data[i]]);
//        for (i64 j = 0; j < nbits; ++j)
//            fprintf(stderr, "%lld ", val[bits[i][j]]);
//        fprintf(stderr, "\n");
//    }
    return bits;
}

vector<vector<i64>> neuralNetwork::bitValueDecomposition(circuit &C, const vector<i64> &data, int nbits, bool has_sign) {
    // bit decomposition
    if (!has_sign) nbits += 1;
    vector<vector<i64>> bits(data.size());
    for (i64 i = 0; i < data.size(); ++i) {
        if (has_sign) {
            bits[i].resize(nbits);
            prepareSignBit(C, data[i], bits[i][0]);
        } else bits[i].resize(nbits);
        for (int j = 1; j < nbits; ++j) {
            prepareBit(C, data[i], bits[i][j], nbits - 1 - j);
        }
    }
    return bits;
}

vector<vector<i64>> neuralNetwork::scaledBits(circuit &C, const vector<vector<i64>> &bits, bool neg) {
    vector<vector<i64>> cbits(bits.size());
    for (i64 i = 0; i < bits.size(); ++i) {
        cbits[i].resize(bits[i].size());
        cbits[i][0] = bits[i][0];
        int n_bits = bits[i].size();
        for (int j = 1; j < n_bits; ++j)
            cbits[i][j] = updateGate(C, Mulc, bits[i][j], !neg ? two_mul[n_bits - 1 - j] : two_mul[Q_BIT_SIZE + n_bits - j]);
    }
    return cbits;
}

vector<vector<vector<i64>>> neuralNetwork::inputLayer(circuit &C) {
    vector<vector<vector<i64>>> res(pic_channel);
    if (mode == "nchw") {
        for (int ci = 0; ci < pic_channel; ++ci) {
            res[ci].resize(pic_size_x);
            for (int x = 0; x < pic_size_x; ++x) {
                res[ci][x].resize(pic_size_y);
                for (int y = 0; y < pic_size_y; ++y) {
                    res[ci][x][y] = updateGate(C, Ins, 0, 0);
                }
            }
        }
    } else {
        for (int ci = 0; ci < pic_channel; ++ci) {
            res[ci].resize(pic_size_x);
            for (int x = 0; x < pic_size_x; ++x) {
                res[ci][x].resize(pic_size_y);
            }
        }
        for (int x = 0; x < pic_size_x; ++x)
            for (int y = 0; y < pic_size_y; ++y)
                for (int ci = 0; ci < pic_channel; ++ci)
                    res[ci][x][y] = updateGate(C, Ins, 0, 0);
    }
    return res;
}

vector<vector<vector<i64>>>
neuralNetwork::naiveConvLayer(circuit &C, const vector<vector<vector<i64>>> &data) {
    // read convolution kernel
    vector<vector<vector<vector<i64>>>> ker(channel_out);
    if (mode == "nchw") {
        for (int i = 0; i < channel_out; ++i) {
            ker[i].resize(channel_in);
            for (int j = 0; j < channel_in; ++j) {
                ker[i][j].resize(m);
                for (int x = 0; x < m; ++x) {
                    ker[i][j][x].resize(m);
                    for (int y = 0; y < m; ++y) {
                        ker[i][j][x][y] = updateGate(C, Ins, 0, 0);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < channel_out; ++i) {
            ker[i].resize(channel_in);
            for (int j = 0; j < channel_in; ++j) {
                ker[i][j].resize(m);
                for (int x = 0; x < m; ++x) {
                    ker[i][j][x].resize(m);
                }
            }
        }
        for (int x = 0; x < m; ++x)
            for (int y = 0; y < m; ++y)
                for (int j = 0; j < channel_in; ++j)
                    for (int i = 0; i < channel_out; ++i)
                            ker[i][j][x][y] = updateGate(C, Ins, 0, 0);

    }
    fprintf(stderr, "kernel size: %lld\n", (i64) channel_out * channel_in * m * m);

    // read convolution bias
    vector<i64> bias(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        bias[co] = updateGate(C, Ins, 0, 0);
        val[bias[co]] <<= SCALE;
        C.gates[bias[co]].u <<= SCALE;
    }

    fprintf(stderr, "bias size: %d\n", channel_out);
    // compute multiplications

    i64 L = -padding, Rx = nx_in + padding, Ry = ny_in + padding;
    vector<vector<vector<vector<i64>>>> res(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        res[co].resize(nx_out);
        for (i64 x = L; x + m <= Rx; x += (1 << log_stride)) {
            res[co][(x - L) >> log_stride].resize(ny_out);
            for (i64 y = L; y + m <= Ry; y += (1 << log_stride)) {
                for (i64 ci = 0; ci < channel_in; ++ci)
                    for (i64 tx = x; tx < x + m; ++tx)
                        for (i64 ty = y; ty < y + m; ++ty)
                            if (check(tx, ty, nx_in, ny_in)) {
                                res[co][(x - L) >> log_stride][(y - L) >> log_stride].push_back(
                                        updateGate(C, Mul, data[ci][tx][ty], ker[co][ci][tx - x][ty - y]));
//                                fprintf(stderr, "%lld ", val.back());
                            }
                res[co][(x - L) >> log_stride][(y - L) >> log_stride].push_back(bias[co]);
//                fprintf(stderr, "\n");
            }
        }
    }

    // compute summation.
    vector<vector<vector<i64>>> new_data(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        new_data[co].resize(nx_out);
        for (int x = 0; x < nx_out; ++x) {
            new_data[co][x].resize(ny_out);
            for (int y = 0; y < ny_out; ++y) {
                new_data[co][x][y] = multiOpt(C, Add, res[co][x][y]);
//                fprintf(stderr, "%lld ", val[new_data[co][x][y]]);
            }
        }
    }
//    fprintf(stderr, "\n");

    fprintf(stderr, "naiveConvLayer: %lu\n", new_data.size());
    return new_data;
}

vector<vector<vector<i64>>> neuralNetwork::reluActConvLayer(circuit &C, const vector<vector<vector<i64>>> &data) {
    // flatten
    auto lst = flatten(data, "");
    auto bits = bitDecomposition(C, lst, Q_MAX, true);
    fprintf(stderr, "relu data in size: %lu\n", lst.size());
    fprintf(stderr, "relu data bits size: %lu * %lu\n", bits.size(), bits[0].size());

    equalCheck(C, lst, bits, true, false);
    bitCheck(C, bits);
    auto sign_rescale = rescaleData(C, bits, true);
    fprintf(stderr, "relu sign rescale size: %lu\n", sign_rescale.size());

    vector<vector<vector<i64>>> new_data(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        new_data[co].resize(nx_out);
        for (int x = 0; x < nx_out; ++x) {
            new_data[co][x].resize(ny_out);
            for (int y = 0; y < ny_out; ++ y) {
                i64 idx = cubIdx(co, x, y, nx_out, ny_out);
                new_data[co][x][y] = sign_rescale[idx];
            }
        }
    }

    fprintf(stderr, "naiveConvLayer: %lu * %lu * %lu\n", new_data.size(), new_data[0].size(), new_data[0][0].size());
    return new_data;
}

vector<i64> neuralNetwork::reluActFconLayer(circuit &C, const vector<i64> &data) {
    auto bits = bitDecomposition(C, data, Q_MAX, true);
    equalCheck(C, data, bits, true, false);
    bitCheck(C, bits);
    auto sign_rescale = rescaleData(C, bits, true);
    return sign_rescale;
}

vector<vector<vector<i64>>>
neuralNetwork::maxPoolingLayer(circuit &C, const vector<vector<vector<i64>>> &data) {
    vector<i64> mx_flatten;
    vector<vector<i64>> data_mx;
    vector<i64> data_mx_flatten;

    fprintf(stderr, "max pooling in size: %lu %lu %lu\n", data.size(), data[0].size(), data[0][0].size());

    // data - max
    for (int co = 0; co < channel_out; ++co) {
        for (int x = 0; x + pool_sz <= nx_out; x += pool_stride)
            for (int y = 0; y + pool_sz <= ny_out; y += pool_stride) {
                i64 mx_dat = 0;
                for (i64 tx = x; tx < x + pool_sz; ++tx)
                    for (i64 ty = y; ty < y + pool_sz; ++ty) {
                        mx_dat = max(mx_dat, val[data[co][tx][ty]]);
                    }

                i64 mx;
                prepareMax(C, mx_dat, mx);
                mx_flatten.push_back(mx);
                i64 neg_mx = updateGate(C, Mulc, mx, -1);

                data_mx.emplace_back();
                for (int tx = x; tx < x + pool_sz; ++tx)
                    for (int ty = y; ty < y + pool_sz; ++ty) {
                        data_mx.back().push_back(updateGate(C, Add, data[co][tx][ty], neg_mx));
                        data_mx_flatten.push_back(data_mx.back().back());
                    }
            }
    }

    fprintf(stderr, "max pooling (data - max) size: %lu %lu\n", data_mx.size(), data_mx[0].size());
    fprintf(stderr, "max pooling (data - max) flatten size: %lu\n", data_mx_flatten.size());

    auto data_mx_bits = bitDecomposition(C, data_mx_flatten, Q_MAX - 1, false);
    fprintf(stderr, "max pooling (data - max) bits size: %lu * %lu\n", data_mx_bits.size(), data_mx_bits[0].size());

    bitCheck(C, data_mx_bits);
    equalCheck(C, data_mx_flatten, data_mx_bits, false, false);

    // check whether zero exists in product of (data - mx)
    vector<i64> zero_exist_check(data_mx.size());
    for (i64 i = 0; i < data_mx.size(); ++i)
        zero_exist_check[i] = multiOpt(C, Mul, data_mx[i], true);
    fprintf(stderr, "max pooling (data - max) zero check size: %lu\n", zero_exist_check.size());

    auto mx_bits = bitDecomposition(C, mx_flatten, Q_MAX - 1, false);
    bitCheck(C, mx_bits);
    equalCheck(C, mx_flatten, mx_bits, false, true);
    auto rescale = rescaleData(C, mx_bits, false);
    fprintf(stderr, "max pooling max size: %lu\n", mx_flatten.size());
    fprintf(stderr, "max pooling max-bits size: %lu\n", mx_bits.size());
    fprintf(stderr, "max pooling max rescale size: %lu\n", rescale.size());
    fprintf(stderr, "max origin: \n");
//    for (auto x: mx_flatten) {
//        fprintf(stderr, "%lld ", val[x]);
//    }
//    fprintf(stderr, "\n");
//
//    fprintf(stderr, "max output: \n");
//    for (auto x: rescale) {
//        fprintf(stderr, "%lld ", val[x]);
//    }
//    fprintf(stderr, "\n");
    vector<vector<vector<i64>>> new_data(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        new_data[co].resize(new_nx_in);
        for (int x = 0; x < new_nx_in; ++x) {
            new_data[co][x].resize(new_ny_in);
            for (int y = 0; y < new_ny_in; ++y) {
                new_data[co][x][y] = rescale[cubIdx(co, x, y, new_nx_in, new_ny_in)];
            }
        }
    }
    fprintf(stderr, "max pooling max result size: %lu * %lu * %lu\n", new_data.size(), new_data[0].size(), new_data[0][0].size());
    return new_data;
}

vector<i64>
neuralNetwork::fullyConnLayer(circuit &C, const vector<i64> &data) {
    fprintf(stderr, "dense in size: %lu\n", data.size());
    vector<vector<i64>> ker(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        ker[co].resize(channel_in);
        for (int ci = 0; ci < channel_in; ++ci) {
            ker[co][ci] = updateGate(C, Ins, 0, 0);
        }
    }
    fprintf(stderr, "dense ker size in size: %lu * %lu\n", ker.size(), ker[0].size());

    vector<i64> bias(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        bias[co] = updateGate(C, Ins, 0, 0);
        val[bias[co]] <<= SCALE;
        C.gates[bias[co]].u <<= SCALE;
    }
    fprintf(stderr, "dense bias size in size: %lu\n", bias.size());

    vector<i64> new_data(channel_out);
    for (int co = 0; co < channel_out; ++co) {
        vector<i64> tmp(channel_in + 1);
        for (int ci = 0; ci < channel_in; ++ci) {
            tmp[ci] = updateGate(C, Mul, data[ci], ker[co][ci]);
        }
        tmp[channel_in] = bias[co];
        new_data[co] = multiOpt(C, Add, tmp);
    }
    fprintf(stderr, "output: \n");
//    for (auto x: new_data) {
//        fprintf(stderr, "%lld ", val[x]);
//    }
    return new_data;
}

void neuralNetwork::refreshConvParam(i64 new_nx, i64 new_ny, i64 new_m, i64 new_chan_in, i64 new_chan_out,
                                     i64 new_log_stride, i64 new_padding) {
    nx_in = new_nx;
    ny_in = new_ny;
    padding = new_padding;
    nx_padded_in = nx_in + (padding * 2);
    ny_padded_in = ny_in + (padding * 2);

    m = new_m;
    channel_in = new_chan_in;
    channel_out = new_chan_out;
    log_stride = new_log_stride;

    nx_out = ((nx_padded_in - m) >> log_stride) + 1;
    ny_out = ((ny_padded_in - m) >> log_stride) + 1;

    new_nx_in = nx_out;
    new_ny_in = ny_out;

}

void neuralNetwork::refreshConvParam(i64 new_nx, i64 new_ny, const convKernel &conv) {
    refreshConvParam(new_nx, new_ny, conv.size, conv.channel_in, conv.channel_out, conv.stride_bl, conv.padding);
}

void neuralNetwork::refreshFCParam(i64 new_chan_in, i64 new_chan_out) {
    nx_in = nx_out = m = 1;
    ny_in = ny_out = 1;
    channel_in = new_chan_in;
    channel_out = new_chan_out;
}

void neuralNetwork::refreshFCParam(const fconKernel &fc) {
    refreshFCParam(fc.channel_in, fc.channel_out);
}

i64 neuralNetwork::getPoolDecmpSize() const {
    switch (pool_ty) {
        case AVG: return new_nx_in * new_ny_in * (pool_bl << 1) * channel_out * pic_parallel;
        case MAX: return new_nx_in * new_ny_in * sqr(pool_sz) * channel_out * pic_parallel * (Q_MAX - 1);
        default:
            assert(false);
    }
}

void neuralNetwork::calcSizeAfterPool(const poolKernel &p) {
    pool_sz = p.size;
    pool_bl = ceilPow2BitLength(pool_sz);
    pool_stride_bl = p.stride_bl;
    pool_stride = 1 << p.stride_bl;
    new_nx_in = ((nx_out - pool_sz) >> pool_stride_bl) + 1;
    new_ny_in = ((ny_out - pool_sz) >> pool_stride_bl) + 1;
}

void neuralNetwork::prepareBit(circuit &C, i64 data, i64 &dcmp_id, i64 bit_shift) {
    if (data < 0) data = -data;
    dcmp_id = updateGate(C, Wit, (data >> bit_shift) & 1, 0);
}

void neuralNetwork::prepareSignBit(circuit &C, i64 data, i64 &dcmp_id) {
    dcmp_id = updateGate(C, Wit, (data < 0), 0);
}

void neuralNetwork::prepareMax(circuit &C, i64 data, i64 &max_id) {
    max_id = updateGate(C, Wit, data, 0);
}