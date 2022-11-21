#include <iostream>

#include "source/tensor.h"
#include "source/helpersp.h"

int main() {
    Tensor x = p::load_npy("data/five.npy",{1,28,28,1});
    x.print_shape(std::cout) << std::endl;
    

    return 0;
}
