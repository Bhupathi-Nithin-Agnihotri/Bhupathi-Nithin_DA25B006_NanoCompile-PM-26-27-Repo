#include "instantiate_graph.hpp"

struct Buffer {
    size_t id;
    double capacity;
};

struct TensorInfo {
    string name;
    double size;
    size_t first;
    size_t last;

    size_t buffer_id;
    string reused_from;
};

struct BufferPlanResult {
    double peak_memory = 0;
    double fragmentation = 0;

    vector<TensorInfo> tensors;
};

template<typename T>
BufferPlanResult plan_memory(const Graph<T>& nn, const unordered_map<string,double>& sizes) {
    using Tptr = unique_ptr<Tensor<T>>;

    const vector<Tptr>& tens = nn.tensors();
    unordered_map<Tensor<T>*, size_t> tens_idx;

    for (size_t i = 0; i < tens.size(); i++) {
        tens_idx[tens[i].get()] = i;
    }
    vector<Operator<T>*> topo = nn.topological_sort();
    unordered_map<Operator<T>*, size_t> op_idx;

    for (size_t i = 0; i < topo.size(); i++) {
        op_idx[topo[i]] = i;
    }

    vector<pair<size_t,size_t>> alive(tens.size(), {SIZE_MAX,SIZE_MAX});

    for (Operator<T>* op : topo) {
        for (Tensor<T>* t : op->inputs()) {
            if (alive[tens_idx[t]].first == SIZE_MAX){
                alive[tens_idx[t]].first = op_idx[op];
            }
            alive[tens_idx[t]].second = op_idx[op];
        }

        for (Tensor<T>* t : op->outputs()) {

            if (alive[tens_idx[t]].first == SIZE_MAX)
                alive[tens_idx[t]].first = op_idx[op];
        }
    }

    vector<TensorInfo> records;

    for (size_t i = 0; i < tens.size(); i++) {
        auto [l,r] = alive[i];
        if (l == SIZE_MAX) {
            continue;
        }
        if (r == SIZE_MAX) {
            r = l;
        }
        TensorInfo info;

        info.name = tens[i]->name();
        info.size = sizes.at(info.name);
        info.first = l;
        info.last = r;
        info.buffer_id = SIZE_MAX;
        info.reused_from = "-";

        if (info.size == 0) {
            continue;
        }
        records.push_back(info);
    }

    sort(records.begin(), records.end(),
        [](const TensorInfo& a, const TensorInfo& b)
        {
            return a.first < b.first;
        }
    );

    struct ActiveBuffer {
        size_t id;
        double capacity;
        size_t release_time;
        string tensor_name;
    };

    struct FreeBuffer {
        size_t id;
        double capacity;
        string previous_tensor;
    };

    vector<ActiveBuffer> active;
    vector<FreeBuffer> free_list;

    size_t next_id = 0;

    double allocated_memory = 0;
    double peak_memory = 0;
    double fragmentation = 0;

    for (TensorInfo& t : records) {
        vector<ActiveBuffer> survivors;
        for (const ActiveBuffer& buf : active) {
            if (buf.release_time < t.first) {
                free_list.push_back({buf.id, buf.capacity, buf.tensor_name});
            }
            else {
                survivors.push_back(buf);
            }
        }

        active.swap(survivors);
        size_t best = -1;

        for (size_t i = 0; i < (size_t)free_list.size(); i++) {
            if (free_list[i].capacity >= t.size) {
                if (best == -1 || free_list[i].capacity < free_list[best].capacity) {
                    best = i;
                }
            }
        }

        if (best != -1) {
            FreeBuffer buf = free_list[best];
            t.buffer_id = buf.id;
            t.reused_from = buf.previous_tensor;
            fragmentation += buf.capacity - t.size;
            active.push_back({buf.id, buf.capacity, t.last, t.name});
            free_list.erase(free_list.begin() + best);
        }
        else {
            size_t id = next_id++;
            t.buffer_id = id;
            allocated_memory += t.size;
            peak_memory = max(peak_memory, allocated_memory);
            active.push_back({id, t.size, t.last, t.name});
        }
    }

    BufferPlanResult result;

    result.peak_memory = peak_memory;
    result.fragmentation = fragmentation;
    result.tensors = move(records);

    return result;
}