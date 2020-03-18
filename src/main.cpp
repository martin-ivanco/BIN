#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <utility>

#include "cartgp/genotype.h"

using namespace std;

namespace node {
    bool OR(const vector<bool> &args) {
        return args[0] || args[1];
    }

    bool XOR(const vector<bool> &args) {
        return args[0] != args[1];
    }

    bool AND(const vector<bool> &args) {
        return args[0] && args[1];
    }

    bool XNOR(const vector<bool> &args) {
        return args[0] == args[1];
    }

    bool IAND(const vector<bool> &args) {
        return (! args[0]) && args[1];
    }
}

namespace compute {
    vector<int> FWHT(const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
        vector<size_t> powers = {1};
        for (int i = 0; i < gt.num_inputs(); i++)
            powers.push_back(powers[i] * 2);
        vector<int> result;
        result.reserve(powers[gt.num_inputs()]);

        // Compute truth table
        vector<bool> inputs(gt.num_inputs(), false);
        result.push_back(gt.evaluate(funcs, inputs)[0]);
        for (int i = 1; i < powers[gt.num_inputs()]; i++) {
            for (int j = 0; j < gt.num_inputs(); j++) {
                if (i % powers[j] == 0)
                    inputs[j] = ! inputs[j];
            }
            result.push_back(gt.evaluate(funcs, inputs)[0]);
        }

        // Compute Fast Walsh-Hadamard Transform
        int x, y;
        for (int i = 0; i < powers.size() - 1; i++) {
            for (int j = 0; j < result.size(); j += powers[i] * 2) {
                for (int k = j; k < j + powers[i]; k++) {
                    x = result[k];
                    y = result[k + powers[i]];
                    result[k] = x + y;
                    result[k + powers[i]] = x - y;
                }
            }
        }

        return result;
    }

    static bool abs_compare(int a, int b) {
        return abs(a) < abs(b);
    }

    bool balance(vector<int> &fwht) {
        return fwht[0] * 2 == fwht.size();
    }

    int non_linearity(vector<int> &fwht) {
        return (fwht.size() - *max_element(fwht.begin(), fwht.end(), abs_compare)) / 2;
    }

    int correlation_immunity(vector<int> &fwht) {
        // Evaluate correlation
        vector<bool> correlation(10, true);
        for (int i = 1; i < fwht.size(); i++) {
            // Count number of 1 bits
            int c = i - ((i >> 1) & 0x55555555);
            c = (c & 0x33333333) + ((c >> 2) & 0x33333333);
            c = ((c + (c >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
            correlation[c - 1] = fwht[i] == 0;
        }

        // Find correlation immunity
        int ci = 0;
        for (int i = 0; i < correlation.size(); i++) {
            if (correlation[i])
                ci++;
            else
                break;
        }

        return ci;
    }
}

namespace fitness {
    double f1(const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
        vector<int> fwht = compute::FWHT(gt, funcs);
        bool balanced = compute::balance(fwht);
        int nf = compute::non_linearity(fwht);
        int ci = compute::correlation_immunity(fwht);
        
        return balanced ? ci * nf : ci * nf / 2;
    }
}

int main(int argc, char **argv) {
    // Set parameters
    cartgp::GeneInt rows = 1; // Generally considered as best practice
    cartgp::GeneInt cols = 1500; // Best results according to paper
    cartgp::GeneInt ins = 10; // Given in assignment
    cartgp::GeneInt outs = 1; // We want just a 1 or 0
    cartgp::GeneInt arity = 2; // Basic node functions have 2 inputs and one output
    cartgp::GeneInt lback = 1500; // Generally considered as best practice to set equal to cols
    vector<cartgp::Function<bool>> funcs = {{"or", node::OR}, {"xor", node::XOR},
                                            {"and", node::AND}, {"xnor", node::XNOR},
                                            {"iand", node::IAND}}; // As in paper
    size_t population = 5; // Size of offspring is therefore 4 - as in paper
    double margin = 1; // Small enough difference between fitness to call evolution stable
    size_t epochs = 500000; // As in paper

    // Construct genotype and evolve it
    cartgp::Genotype gt(arity, funcs.size(), lback, ins, outs, rows, cols);
    auto results = gt.evolve(funcs, population - 1, margin, epochs, {fitness::f1});
}
