#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>
#include <utility>
#include <chrono>
#include <thread>

#include "cartgp/genotype.h"

#ifdef _OPENMP
#include <omp.h>
#endif

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
            result.push_back(gt.evaluate(funcs, inputs)[0] ? 1 : -1);
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
        return fwht[0] == 0;
    }

    int non_linearity(vector<int> &fwht) {
        return (fwht.size() - abs(*max_element(fwht.begin(), fwht.end(), abs_compare))) / 2;
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
    // All scores are based on non-linearity - that is the main objective
    // Balancedness and correlation immunity pose as rewards or penalties

    tuple<double, bool, int, int> f1(
            const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
        vector<int> fwht = compute::FWHT(gt, funcs);
        bool b = compute::balance(fwht);
        int nf = compute::non_linearity(fwht);
        int ci = compute::correlation_immunity(fwht, gt.num_inputs());
        // If the function is balanced, score gets doubled
        double score = b ? nf * 2 : nf;
        // If the function has correlation immunity of at least 1, score gets doubled
        score = ci > 0 ? score * 2 : score;
        return {score, b, nf, ci};
    }

    tuple<double, bool, int, int> f2(
            const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
        vector<int> fwht = compute::FWHT(gt, funcs);
        bool b = compute::balance(fwht);
        int nf = compute::non_linearity(fwht);
        int ci = compute::correlation_immunity(fwht, gt.num_inputs());
        // If the function is balanced and has correlation immunity equal to 1
        // the score is quadrupled - strong emphasis on not having higher CI
        double score = nf;
        if (b && (ci == 1))
            score *= 4;
        return {score, b, nf, ci};
    }

    tuple<double, bool, int, int> f3(
            const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
        vector<int> fwht = compute::FWHT(gt, funcs);
        bool b = compute::balance(fwht);
        int nf = compute::non_linearity(fwht);
        int ci = compute::correlation_immunity(fwht, gt.num_inputs());
        // If the function is balanced, the score is multiplied by 1 + 1/5 CI
        double score = b ? (1 + static_cast<double>(ci) / 5) * nf : nf;
        return {score, b, nf, ci};
    }
}

string epoch2csv(tuple<double, bool, int, int> e) {
    return to_string(get<0>(e)) + "," + to_string(get<1>(e)) + "," + to_string(get<2>(e)) + ","
           + to_string(get<3>(e));
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
    if ((argc != 2) && (argc != 3)) {
        cerr << "usage: ./evobool <fitness_func> [<num_threads>]" << endl;
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
    size_t epochs = 500000; // As in paper

    // Set fitness function
    cartgp::FitnessFunction<bool> ff;
    if (string(argv[1]) == string("1"))
        ff = {fitness::f1};
    else if (string(argv[1]) == string("2"))
        ff = {fitness::f2};
    else if (string(argv[1]) == string("3"))
        ff = {fitness::f3};
    else {
        cerr << "ERROR: Invalid fitness function." << endl;
        return 1;
    }

    // Set number of threads (if OpenMP is used)
#ifdef _OPENMP
    if (argc == 3)
        omp_set_num_threads(atoi(argv[2]));
    else
        omp_set_num_threads(1);
#endif

    // Running evolution 100 times
#pragma omp parallel for
    for (int i = 0; i < 100; i++) {
        // Sleep a bit just in case random number generator is based on time
        this_thread::sleep_for(chrono::milliseconds(i));

        // Construct genotype and evolve it
        cartgp::Genotype gt(arity, funcs.size(), lback, ins, outs, rows, cols);
        vector<tuple<double, bool, int, int>> ed;
        auto [evolved, solution] = gt.evolve(funcs, population - 1, epochs, ff, ed);

        // Store results
        ofstream csv("output/result" + to_string(i) + ".csv");
        csv << "Epoch,Fitness,Balanced,Non-linearity,Correlation immunity" << endl;
        for (int j = 0; j < ed.size(); j++)
            csv << to_string(j) + "," + epoch2csv(ed[j]) << endl;
        save_table(evolved, funcs, "output/table" + to_string(i));
    }
}
