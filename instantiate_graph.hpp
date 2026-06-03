#pragma once
#include "ops.hpp"

template <typename T>
unique_ptr<Graph<T>> instantiate_graph(const vector<string>& graph_input) {
    auto nn = make_unique<Graph<T>>();

    unordered_map<string, unique_ptr<Tensor<T>>> tensors;

    for (const string& line : graph_input) {

        size_t colon = line.find(':');
        size_t lpar  = line.find('(');
        size_t rpar  = line.find(')');
        size_t arrow = line.find("->");

        string node_name = line.substr(0, colon);

        string op_type =
            line.substr(colon + 1, lpar - colon - 1);

        op_type.erase(
            remove(op_type.begin(), op_type.end(), ' '),
            op_type.end());

        string args =
            line.substr(lpar + 1, rpar - lpar - 1);

        string output =
            line.substr(arrow + 2);

        output.erase(
            remove(output.begin(), output.end(), ' '),
            output.end());

        vector<string> inputs;

        stringstream ss(args);
        string token;

        while (getline(ss, token, ',')) {
            token.erase(
                remove(token.begin(), token.end(), ' '),
                token.end());
            inputs.push_back(token);
        }

        for (const auto& name : inputs) {
            if (!tensors.count(name)) {
                tensors[name] =
                    make_unique<Tensor<T>>(
                        vector<size_t>{1});
                tensors[name]->change_name(name);
            }
        }

        if (!tensors.count(output)) {
            tensors[output] =
                make_unique<Tensor<T>>(
                    vector<size_t>{1});
            tensors[output]->change_name(output);
        }

        unique_ptr<Operator<T>> op;

        if (op_type == "Conv") {
            op = make_unique<Conv<T>>(1);
        }
        else if (op_type == "ReLU") {
            op = make_unique<ReLU<T>>(1);
        }
        else if (op_type == "Add") {
            op = make_unique<Add<T>>(1);
        }
        else if (op_type == "Mul") {
            op = make_unique<Mul<T>>(1);
        }
        else {
            throw runtime_error(
                "Unknown operator: " + op_type);
        }

        op->change_name(node_name);

        for (const auto& inp : inputs) {
            op->add_input(tensors[inp].get());
        }

        op->add_output(tensors[output].get());

        nn->add_op(move(op));
    }

    for (auto& [name, tensor] : tensors) {
        nn->add_tensor(move(tensor));
    }

    return nn;
}
