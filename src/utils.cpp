//
// Created by 69029 on 3/9/2021.
//

#include <cmath>
#include <iostream>
#include <cassert>
#include "utils.hpp"

using std::cerr;
using std::endl;
using std::string;
using std::cin;

int ceilPow2BitLength(i64 n) {
    return n < 1e-9 ? -1 : (int) ceil(log(n) / log(2.));
}

bool check(i64 x, i64 y, i64 nx, i64 ny) {
    return 0 <= x && x < nx && 0 <= y && y < ny;
}

i64 sqr(i64 x) {
    return x * x;
}

i64 matIdx(i64 x, i64 y, i64 n) {
    assert(y < n);
    return x * n + y;
}

i64 cubIdx(i64 x, i64 y, i64 z, i64 n, i64 m) {
    assert(y < n && z < m);
    return matIdx(matIdx(x, y, n), z, m);
}

i64 tesIdx(i64 w, i64 x, i64 y, i64 z, i64 n, i64 m, i64 l) {
    assert(x < n && y < m && z < l);
    return matIdx(cubIdx(w, x, y, n, m), z, l);
}


