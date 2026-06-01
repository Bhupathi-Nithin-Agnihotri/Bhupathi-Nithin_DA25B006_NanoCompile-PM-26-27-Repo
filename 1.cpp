#include "1.hpp"
template<typename T>
class Add : protected Operator<T> {
    
};

template<typename T>
class Mul : protected Operator<T> {
};

template<typename T>
class ReLU : protected Operator<T> {

};

template<typename T>
class Conv : protected Operator<T> {

};



int main() {
    return 0;
}