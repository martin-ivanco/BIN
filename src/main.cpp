#include <iostream>
#include <vector>
#include <utility>

#include "cartgp/genotype.h"

using namespace std;

bool f_or(const vector<bool> &args) {
    return args[0] || args[1];
}

bool f_xor(const vector<bool> &args) {
    return args[0] != args[1];
}

bool f_and(const vector<bool> &args) {
    return args[0] && args[1];
}

bool f_xnor(const vector<bool> &args) {
    return args[0] == args[1];
}

bool f_iand(const vector<bool> &args) {
    return (! args[0]) && args[1];
}

double fitness1(const cartgp::Genotype &gt, const vector<cartgp::Function<bool>> &funcs) {
    return 0;
}

int main(int argc, char **argv) {
    // Set parameters
    cartgp::GeneInt rows = 1; // Generally considered as best practice
    cartgp::GeneInt cols = 1500; // Best results according to paper
    cartgp::GeneInt ins = 10; // Given in assignment
    cartgp::GeneInt outs = 1; // Boolean functions have 1 output
    cartgp::GeneInt arity = 2; // Basic node functions have 2 inputs
    cartgp::GeneInt lback = 1500; // Generally considered as best practice to set equal to cols
    vector<cartgp::Function<bool>> funcs = {{"or", f_or}, {"xor", f_xor}, {"and", f_and},
                                            {"xnor", f_xnor}, {"iand", f_iand}}; // As in paper
    size_t population = 5; // Size of offspring is therefore 4 - as in paper
    double margin = 0.000001; // Small enough difference between fitness to call evolution stable
    size_t epochs = 500000; // As in paper

    // Construct genotype and evolve it
    cartgp::Genotype gt(arity, funcs.size(), lback, ins, outs, rows, cols);
    auto results = gt.evolve(funcs, population - 1, margin, epochs, {fitness1});
}
