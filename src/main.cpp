#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
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

    int correlation_immunity(vector<int> &fwht, int cap) {
        // Evaluate correlation
        vector<bool> correlation(cap, true);
        for (int i = 1; i < fwht.size(); i++) {
            // Count number of 1 bits
            int c = i - ((i >> 1) & 0x55555555);
            c = (c & 0x33333333) + ((c >> 2) & 0x33333333);
            c = ((c + (c >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
            if (correlation[c - 1])
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
        int score = compute::non_linearity(fwht);
        score = compute::balance(fwht) ? score * 4 : score;
        score = compute::correlation_immunity(fwht, gt.num_inputs()) ? score * 2 : score;
        return score;
    }
}

string results2csv(cartgp::Genotype &gt, cartgp::SolutionInfo &si,
                   vector<cartgp::Function<bool>> &funcs) {
    vector<int> fwht = compute::FWHT(gt, funcs);
    return to_string(si.steps) + "," + to_string(si.fitness) + ","
           + to_string(compute::balance(fwht)) + "," + to_string(compute::non_linearity(fwht))
           + "," + to_string(compute::correlation_immunity(fwht, gt.num_inputs()));
}

void save_table(cartgp::Genotype &gt, vector<cartgp::Function<bool>> &funcs, string fname) {
    // Count powers
    vector<size_t> powers = {1};
    for (int i = 0; i < gt.num_inputs(); i++)
        powers.push_back(powers[i] * 2);

    // Open file and write values
    ofstream file(fname, ios::binary);
    char eight_bits = 0;
    vector<bool> inputs(gt.num_inputs(), false);
    eight_bits += gt.evaluate(funcs, inputs)[0] ? 1 : 0;
    for (int i = 1; i < powers[gt.num_inputs()]; i++) {
        for (int j = 0; j < gt.num_inputs(); j++) {
            if (i % powers[j] == 0)
                inputs[j] = ! inputs[j];
        }
        eight_bits <<= 1;
        eight_bits += gt.evaluate(funcs, inputs)[0] ? 1 : 0;
        if ((i + 1) % 8 == 0)
            file.write(&eight_bits, sizeof(eight_bits));
    }
}

int main(int argc, char **argv) {
    // Check args
    if (argc != 2) {
        cerr << "usage: ./evobool <fitness_function>" << endl;
        return 1;
    }

    // Set static parameters
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

    // Set fitness function
    cartgp::FitnessFunction<bool> ff;
    if (string(argv[1]) == string("1"))
        ff = {fitness::f1};
    else {
        cerr << "ERROR: Invalid fitness function." << endl;
        return 1;
    }

    // Prepare csv
    ofstream csv("output/results.csv");
    csv << "Run,Epochs,Fitness,Balanced,Non-linearity,Correlation immunity" << endl;

    // Running evolution 100 times
    for (int i = 0; i < 100; i++) {
        // Construct genotype and evolve it
        cartgp::Genotype gt(arity, funcs.size(), lback, ins, outs, rows, cols);
        auto [evolved, solution] = gt.evolve(funcs, population - 1, margin, epochs, ff);
        csv << to_string(i) + "," + results2csv(evolved, solution, funcs) << endl;
        save_table(evolved, funcs, "output/table" + to_string(i));
    }
}
