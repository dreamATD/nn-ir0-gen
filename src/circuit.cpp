#include "circuit.h"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <cassert>

using std::ofstream;

void circuit::init(int q_bit_size) {
    two_mul.resize((q_bit_size + 1) << 1);
    two_mul[0] = 1;
    two_mul[q_bit_size + 1] = -1;
    for (int i = 1; i <= q_bit_size; ++i) {
        two_mul[i] = two_mul[i - 1] + two_mul[i - 1];
        two_mul[i + q_bit_size + 1] = -two_mul[i];
    }
}

void circuit::print(char *ins_file, char *wit_file, char *rel_file) {
    ofstream ins(ins_file), wit(wit_file), rel(rel_file);
    rel << "// Header start\n"
        << "version 1.0.0;\n"
        << "field characteristic 2305843009213693951 degree 1;\n"
        << "relation\n"
        << "gate_set: boolean;\n"
        << "features: simple;\n"
        << "@begin\n";
    ins << "version 1.0.0;\n"
        << "field characteristic 2305843009213693951 degree 1;\n"
        << "instance\n"
        << "@begin\n";
    wit << "version 1.0.0;\n"
        << "field characteristic 2305843009213693951 degree 1;\n"
        << "short_witness\n"
        << "@begin\n";

    fprintf(stderr, "the number of gates: %lu\n", gates.size());
    
    u64 n_add_gates = 0, n_addc_gates = 0, n_mul_gates = 0, n_mulc_gates = 0;
    u64 n_ins_gates = 0, n_wit_gates = 0;
    for (i64 g = 0; g < gates.size(); ++g) {
        auto &gate = gates[g];
        switch (gate.ty) {
            case Ins:
                ins << string_format("< %lld >;\n", gate.u);
                rel << string_format("$%lld<-@instance;\n", g);
                ++n_ins_gates;
                break;
            case Wit:
                wit << string_format("< %lld >;\n", gate.u);
                rel << string_format("$%lld<-@short_witness;\n", g);
                ++n_wit_gates;
                break;
            case Add:
                rel << string_format("$%lld <- @add($%lld,$%lld);\n", g, gate.u, gate.v);
                ++n_add_gates;
                break;
            case Mul:
                rel << string_format("$%lld <- @mul($%lld,$%lld);\n", g, gate.u, gate.v);
                ++n_mul_gates;
                break;
            case Addc:
                rel << string_format("$%lld <- @addc($%lld,< %lld >);\n", g, gate.u, gate.v);
                ++n_addc_gates;
                break;
            case Mulc:
                rel << string_format("$%lld <- @mulc($%lld,< %lld >);\n", g, gate.u, gate.v);
                ++n_mulc_gates;
                break;
            case Xor:
                rel << string_format("$%lld <- @xor($%lld,$%lld);\n", g, gate.u, gate.v);
                break;
            case And:
                rel << string_format("$%lld <- @and($%lld,$%lld);\n", g, gate.u, gate.v);
                break;
            case Not:
                rel << string_format("$%lld <- @not($%lld);\n", g, gate.u, gate.v);
                break;
            default:
                assert(false);
        }
    }
    ins << "@end\n";
    wit << "@end\n";
    rel << "@end\n";
    ins.close();
    wit.close();
    rel.close();

    fprintf(stderr, "# add  = %llu\n", n_add_gates);
    fprintf(stderr, "# mul  = %llu\n", n_mul_gates);
    fprintf(stderr, "# addc = %llu\n", n_addc_gates);
    fprintf(stderr, "# mulc = %llu\n", n_mulc_gates);
    fprintf(stderr, "# ins  = %llu\n", n_ins_gates);
    fprintf(stderr, "# wit  = %llu\n", n_wit_gates);
}
