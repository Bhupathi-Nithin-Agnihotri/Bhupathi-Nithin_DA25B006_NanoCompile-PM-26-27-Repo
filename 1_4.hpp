# pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <memory>

using namespace std;

template<typename T>
class Operator;

template<typename T>
class Tensor {
public:
    using dtype = T;

    Tensor() = default;

    Tensor(const vector<size_t>& shape) : shape_(shape) {
        data_.resize(num_ele());
    }

    Tensor(const vector<size_t>& shape, const vector<T>& data) : shape_(shape), data_(data) {
        if (num_ele() != data_.size()) {
            throw invalid_argument("Shape and data size don't match");
        }
    }

    Tensor(vector<size_t>&& shape, vector<T>&& data) : shape_(move(shape)), data_(move(data)) {
        if (num_ele() != data_.size()) {
            throw invalid_argument("Shape and data size don't match");
        }
    }

    const vector<size_t>& shape() const {
        return shape_;
    }

    vector<T>& data() {
        return data_;
    }

    const vector<T>& data() const {
        return data_;
    }

    size_t num_ele() const {
        size_t ans = 1;

        for (size_t dim : shape_) {
            ans *= dim;
        }

        return ans;
    }

    template<typename... Indices>
    T& operator()(Indices... indices) {
        vector<size_t> idx{
            static_cast<size_t>(indices)...
        };
        size_t curr = 0;
        size_t jump = 1;

        for (size_t i = shape_.size(); i-- > 0;) {

            if (idx[i] >= shape_[i]) {
                throw out_of_range("Out of bounds");
            }

            curr += idx[i] * jump;
            jump *= shape_[i];
        }
        return data_[curr];
    }

    template<typename... Indices>
    const T& operator()(Indices... indices) const {
        vector<size_t> idx{
            static_cast<size_t>(indices)...
        };
        size_t curr = 0;
        size_t jump = 1;

        for (size_t i = shape_.size(); i-- > 0;) {

            if (idx[i] >= shape_[i]) {
                throw out_of_range("Out of bounds");
            }

            curr += idx[i] * jump;
            jump *= shape_[i];
        }

        return data_[curr];
    }

    const Operator<T>* producer() const {
        return producer_;
    }

    vector<const Operator<T>*> consumers() const {
        vector<const Operator<T>*> result;

        result.reserve(consumers_.size());
        for (auto* op : consumers_) {
            result.push_back(op);
        }

        return result;
    }

    size_t num_consumers() const {
        return consumers_.size();
    }

private:
    vector<size_t> shape_;
    vector<T> data_;

    Operator<T>* producer_ = nullptr;
    vector<Operator<T>*> consumers_;

    friend class Operator<T>;
};

template<typename T>
class Operator {
public:
    virtual ~Operator() = default;
    virtual string name() const = 0;
    virtual void calc() = 0;


    const vector<Tensor<T>*>& inputs() const {
        return inputs_;
    }

    const vector<Tensor<T>*>& outputs() const {
        return outputs_;
    }

    void add_input(Tensor<T>* tensor) {
        inputs_.push_back(tensor);
        tensor->consumers_.push_back(this);
    }

    void add_output(Tensor<T>* tensor) {
        outputs_.push_back(tensor);
        tensor->producer_ = this;
    }

protected:
    vector<Tensor<T>*> inputs_;
    vector<Tensor<T>*> outputs_;
};
template<typename T>
class Graph {
public:
    using Tptr = unique_ptr<Tensor<T>>;
    using Optr = unique_ptr<Operator<T>>;

    Graph() = default;
    ~Graph() = default;
    Graph(vector<Optr> operators, vector<Tptr> tensors) : operators_(move(operators)), tensors_(move(tensors)) {};
    void add_op(Optr op) {
        operators_.push_back(move(op));
    }
    void add_tensor(Tptr tensor) {
        tensors_.push_back(move(tensor));
    }
    bool is_valid() {

    }
private:
    vector<Optr> operators_;
    vector<Tptr> tensors_;
};
