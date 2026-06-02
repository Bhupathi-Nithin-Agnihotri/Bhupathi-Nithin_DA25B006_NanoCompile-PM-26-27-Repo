#include <algorithm>
#include "1.hpp"

template<typename T>
class Add : public Operator<T> {
public:
    Add(double x) { this->change_cost(x); }
    string name() const override {
        return "Add";
    }

    void calc() override {
        if (this->inputs_.size() != 2 || this->outputs_.size() != 1)
            throw invalid_argument("Add requires 2 inputs and 1 output");

        if (this->inputs_[0]->shape() != this->inputs_[1]->shape() || this->outputs_[0]->shape() != this->inputs_[0]->shape()) {
            throw invalid_argument("Shape mismatch");
        }
        size_t n = this->inputs_[0]->num_ele();
        auto* a = this->inputs_[0];
        auto* b = this->inputs_[1];
        auto* out = this->outputs_[0];
        for (size_t i = 0; i < n; i++) {
            (*out)(i) = (*a)(i) + (*b)(i);
        }
    }
};

template<typename T>
class Mul : public Operator<T> {
public:
    Mul(double x) { this->change_cost(x); }
    string name() const override {
        return "Mul";
    }

    void calc() override {
        if (this->inputs_.size() != 2 || this->outputs_.size() != 1) {
            throw invalid_argument("Mul requires 2 inputs and 1 output");
        }

        if (this->inputs_[1]->shape().size() != 1 || this->outputs_[0]->shape() != this->inputs_[0]->shape()) {
            throw invalid_argument("Shape mismatch");
        }
        size_t n = this->inputs_[0]->num_ele();
        auto* a = this->inputs_[0];
        auto* b = this->inputs_[1];
        auto* out = this->outputs_[0];
        for (size_t i = 0; i < n; i++) {
            (*out)(i) = (*a)(i) * (*b)();
        }
    }
};

template<typename T>
class ReLU : public Operator<T> {
public:
    ReLU(double x) { this->change_cost(x); }
    string name() const override {
        return "ReLU";
    }

    void calc() override {
        if (this->inputs_.size() != 1 || this->outputs_.size() != 1)
            throw invalid_argument("ReLU requires 1 input and 1 output");

        if (this->inputs_[0]->shape() != this->outputs_[0]->shape())
            throw invalid_argument("Shape mismatch");

        size_t n = this->inputs_[0]->num_ele();
        auto* a = this->inputs_[0];
        auto* out = this->outputs_[0];
        for (size_t i = 0; i < n; i++) {
            (*out)(i) = max((*a)(i), T(0));
        }
    }
};

template<typename T>
class Conv : public Operator<T> {
public:
    Conv(double x) { this->change_cost(x); }
    string name() const override {
        return "Conv";
    }

    void calc() override {
        if (this->inputs_.size() != 2 || this->outputs_.size() != 1) {
            throw invalid_argument("Conv requires input tensor, kernel tensor and output tensor");
        }

        auto* input  = this->inputs_[0];
        auto* kernel = this->inputs_[1];
        auto* output = this->outputs_[0];

        if (input->shape().size() != 1 || kernel->shape().size() != 1 || output->shape().size() != 1) {
            throw invalid_argument("Only 1D convolution supported T-T");
        }

        size_t n = input->shape()[0];
        size_t k = kernel->shape()[0];

        if (output->shape()[0] != n - k + 1) {
            throw invalid_argument("Output shape incorrect");
        }

        for (size_t i = 0; i + k <= n; i++) {
            T sum{};
            for (size_t j = 0; j < k; j++) {
                sum += input->data()[i + j] * kernel->data()[j];
            }
            output->data()[i] = sum;
        }
    }
};


int main() {
    vector<string> graph_input = {
        "n1: Conv(x, w1) -> a",
        "n2: ReLU(a) -> b",
        "n3: Conv(b, w2) -> c",
        "n4: Conv(x, w_skip) -> s",
        "n5: Add(c, s) -> d",
        "n6: ReLU(d) -> e",
        "n7: Mul(e, scale) -> f",
        "n8: Add(f, bias) -> y"
    };

    unordered_map<string, double> costs = {
        {"n1", 8},
        {"n3", 10},
        {"n4", 5}
    };

    Graph<double> nn;
    unordered_map<string, unique_ptr<Tensor<double>>> tensors;

    for (string line : graph_input) {
        size_t colon = line.find(':');
        size_t lpar  = line.find('(');
        size_t rpar  = line.find(')');
        size_t arrow = line.find("->");

        string node_name = line.substr(0, colon);
        string op_type = line.substr(colon + 1, lpar - colon - 1);
        op_type.erase(remove(op_type.begin(), op_type.end(), ' '), op_type.end());
        string args      = line.substr(lpar + 1, rpar - lpar - 1);
        string output = line.substr(arrow + 2);
        output.erase(remove(output.begin(), output.end(), ' '), output.end());

        vector<string> inputs;

        string token;
        stringstream ss(args);

        while (getline(ss, token, ',')) {
            token.erase(remove(token.begin(), token.end(), ' '), token.end());
            inputs.push_back(token);
        }

        for (auto &name : inputs) {
            if (!tensors.count(name)) {
                tensors[name] =
                    make_unique<Tensor<double>>(vector<size_t>{1});
            }
        }

        if (!tensors.count(output)) {
            tensors[output] =
                make_unique<Tensor<double>>(vector<size_t>{1});
        }

        unique_ptr<Operator<double>> op;

        if (op_type == "Conv") {
            op = make_unique<Conv<double>>(1);
        }
        else if (op_type == "ReLU") {
            op = make_unique<ReLU<double>>(1);
        }
        else if (op_type == "Add") {
            op = make_unique<Add<double>>(1);
        }
        else if (op_type == "Mul") {
            op = make_unique<Mul<double>>(1);
        }
        else {
            throw runtime_error("Unknown operator: " + op_type);
        }

        if (costs.count(node_name)) {
            op->change_cost(costs[node_name]);
        }

        for (auto &inp : inputs) {
            op->add_input(tensors[inp].get());
        }

        op->add_output(tensors[output].get());

        nn.add_op(move(op));
    }

    for (auto &[name, tensor] : tensors) {
        nn.add_tensor(move(tensor));
    }
    cout << nn.get_fastest_execution() << "\n";
    nn.print_graph();
    
   return 0;
}