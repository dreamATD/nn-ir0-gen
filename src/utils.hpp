//
// Created by 69029 on 3/9/2021.
//

#ifndef ZKCNN_UTILS_HPP
#define ZKCNN_UTILS_HPP

#include "circuit.h"
#include "typedef.hpp"
#include <string>
using std::string;

int ceilPow2BitLength(i64 n);

bool check(i64 x, i64 y, i64 nx, i64 ny);

i64 matIdx(i64 x, i64 y, i64 n);

i64 cubIdx(i64 x, i64 y, i64 z, i64 n, i64 m);

i64 tesIdx(i64 w, i64 x, i64 y, i64 z, i64 n, i64 m, i64 l);

i64 sqr(i64 x);

template<typename ... Args>
string string_format(const string& format, Args ... args){
    size_t size = 1 + snprintf(nullptr, 0, format.c_str(), args ...);  // Extra space for \0
    // unique_ptr<char[]> buf(new char[size]);
    char bytes[size];
    snprintf(bytes, size, format.c_str(), args ...);
    return string(bytes);
}

u64 convert2Unsigned(i64 x);

#endif //ZKCNN_UTILS_HPP
