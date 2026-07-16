#pragma once
#include <string>
#include <vector>
#include <utec/algebra/tensor.h>

namespace utec::utils {

struct Dataset {
    utec::algebra::Tensor<float, 2> features;
    utec::algebra::Tensor<float, 2> labels;
};

class CSVLoader {
public:
    // Carga un dataset CSV. max_rows limita la cantidad de datos para pruebas rápidas.
    static Dataset load_credit_card_data(const std::string& filepath, size_t max_rows = 0);
};

}
