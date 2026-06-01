# pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <utility>
#include <memory>
#include <unordered_map>
#include <queue>
#include <iostream>
#include <sstream>

using namespace std;

template<typename T>
class Operator;

template<typename T>
class Graph;

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
    friend class Graph<T>;
};

template<typename T>
class Operator {
public:
    virtual ~Operator() = default;
    virtual string name() const = 0;
    virtual void calc() = 0;
    virtual double cost() const = 0;

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

    vector<Operator<T>*> topological_sort() const {
        unordered_map<Operator<T>*, int> in_degree;
        unordered_map<Operator<T>*, vector<Operator<T>*>> adj;

        for (auto& op_ptr : operators_) {
            Operator<T>* op = op_ptr.get();
            if (!in_degree.count(op)) {
                in_degree[op] = 0;
            }

            for (Tensor<T>* out : op->outputs()) {
                for (Operator<T>* consumer : out->consumers_) {
                    adj[op].push_back(consumer);
                    in_degree[consumer]++;
                }
            }
        }
        queue<Operator<T>*> ready;
        for (auto& [op, deg] : in_degree) {
            if (deg == 0) {
                ready.push(op);
            }
        }

        vector<Operator<T>*> res;
        res.reserve(operators_.size());
        while (!ready.empty()) {
            Operator<T>* cur = ready.front();
            ready.pop();
            res.push_back(cur);

            for (Operator<T>* succ : adj[cur]) {
                if (--in_degree[succ] == 0) {
                    ready.push(succ);
                }
            }
        }

        if (res.size() != operators_.size()) {
            throw runtime_error("Cycle Detected.");
        }
        return res;
    }

    void forward() {
        for (Operator<T>* op : topological_sort()) {
            op->calc();
        }
    }

    void print_graph() const {
        //doubt
    }

    double get_fastest_execution() const {
        vector<Operator<T>*> order = topological_sort();
        unordered_map<Operator<T>*, double> earliest;

        for (Operator<T>* op : order) {
            double start = 0.0;
            for (Tensor<T>* in : op->inputs()) {
                Operator<T>* producer = const_cast<Operator<T>*>(in->producer());

                if (producer) {
                    start = max(start, earliest[producer] + producer->cost());
                }
            }
            earliest[op] = start;
        }

        double ans = 0.0;
        for (auto& [op, t] : earliest) {
            ans = max(ans, t + op->cost());
        }
        return ans;
    }
private:
    vector<Optr> operators_;
    vector<Tptr> tensors_;
};
