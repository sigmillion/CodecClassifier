#include <numeric>      // std::iota
#include <algorithm>    // std::sort
#include <iostream>     // std::cout
#include <functional>   // std::bind
using namespace std::placeholders;

double data[10]={2.7182, 3.14159, 1.202, 1.618, 0.5772, 1.3035, 2.6854, 1.32471, 0.70258, 4.6692};
int index[10];

bool compare(int a, int b, double* data)
{
    return data[a]<data[b];
}

int main()
{
    std::iota(std::begin(index), std::end(index), 0); // fill index with {0,1,2,...} This only needs to happen once

    std::sort(std::begin(index), std::end(index), std::bind(compare,  _1, _2, data ));

    std::cout << "data:   "; for (int i=0; i<10; i++) std::cout << data[i] << " "; std::cout << "\n";
    std::cout << "index:  "; for (int i=0; i<10; i++) std::cout << index[i] << " "; std::cout << "\n";
    std::cout << "sorted: "; for (int i=0; i<10; i++) std::cout << data[index[i]] << " "; std::cout << "\n";

    std::cin.get();
    return 0;
}
