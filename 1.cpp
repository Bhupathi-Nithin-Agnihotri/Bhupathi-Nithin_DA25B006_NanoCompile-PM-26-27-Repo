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
            out[i] = a[i] + b[i];
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

        if (this->inputs_[0]->shape() != this->inputs_[1]->shape() || this->outputs_[0]->shape() != this->inputs_[0]->shape()) {
            throw invalid_argument("Shape mismatch");
        }
        size_t n = this->inputs_[0]->num_ele();
        auto* a = this->inputs_[0];
        auto* b = this->inputs_[1];
        auto* out = this->outputs_[0];
        for (size_t i = 0; i < n; i++) {
            out[i] = a[i] * b[i];
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
            out[i] = max(ta->data()[i], T(0));
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
    return 0;
}