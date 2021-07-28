#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <unordered_set>
#include <iostream>
#include "typedef.hpp"

using std::cerr;
using std::endl;
using std::vector;

enum GateType {
    Add, Mul, Addc, Mulc, Xor, And, Not, Ins, Wit
};

struct Gate {
    GateType ty;
    i64 u, v;
    i64 global_id;
    bool is_assert;

    Gate(GateType _ty = Wit, i64 _u = 0, i64 _v = 0, bool _is_assert = false):
        ty(_ty), u(_u), v(_v) {
        is_assert = _is_assert;
    }
};

class layeredCircuit {
public:
    vector<Gate> gates;
    vector<i64> two_mul;

    void init(int q_bit_size);

    void print(char *ins_file, char *wit_file, char *rel_file);
};

