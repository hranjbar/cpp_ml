#include <iostream>
#include "cppflow/cppflow.h"


int main() {

    auto input = cppflow::fill({10, 5}, 1.0f);
    cppflow::model model("../mnist_tf/saved_model");
    auto output = model(input);

    std::cout << output << std::endl;

    return 0;
}
