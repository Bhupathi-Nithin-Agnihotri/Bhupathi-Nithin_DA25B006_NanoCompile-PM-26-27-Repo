#include "3.hpp"

int main() {
    vector<string> graph_input = {
        "n1: Conv(X, W1) -> A",
        "n2: ReLU(A) -> B",
        "n3: Conv(B, W2) -> C",
        "n4: Conv(X, W3) -> S",
        "n5: Add(C, S) -> D",
        "n6: ReLU(D) -> E",
        "n7: Conv(E, W4) -> F",
        "n8: Add(F, B) -> G",
        "n9: ReLU(G) -> Y"
    };

    unique_ptr<Graph<double>> nn = instantiate_graph<double>(graph_input);
    
    unordered_map<string, double> sizes = {
        {"A", 24},
        {"B", 24},
        {"C", 32},
        {"D", 32},
        {"E", 32},
        {"F", 24},
        {"G", 24},
        {"Y", 24},
        {"S", 32},

        {"X", 0},
        {"W1", 0},
        {"W2", 0},
        {"W3", 0},
        {"W4", 0}
    };
    BufferPlanResult res =
    plan_memory(*(nn.get()), sizes);

    cout << "Peak Memory = " << res.peak_memory << " KB\n";

    cout << "Fragmentation = " << res.fragmentation << " KB\n";

    for (const TensorInfo& t : res.tensors) {
        cout << t.name << " size: " << t.size;
        cout << " first: " << t.first;
        cout << " last: " << t.last;
        cout << " buffer: " << t.buffer_id;
        cout << " reused_from: " << t.reused_from;
        cout << "\n";
    }
}