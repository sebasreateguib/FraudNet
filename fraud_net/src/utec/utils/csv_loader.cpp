#include "utec/utils/csv_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>

namespace utec::utils {

Dataset CSVLoader::load_credit_card_data(const std::string& filepath, size_t max_rows) {
    std::cout << "Cargando dataset desde " << filepath << "...\n";
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filepath);
    }
    
    std::string line;
    // Ignorar la primera línea (el encabezado)
    std::getline(file, line);
    
    std::vector<std::vector<float>> all_features;
    std::vector<float> all_labels;
    
    auto parse_token = [](const std::string& t) {
        std::string clean;
        for (char c : t) {
            if (c != '"' && c != '\r' && c != '\n') {
                clean += c;
            }
        }
        return std::stof(clean);
    };
    
    int count_fraud = 0;
    int count_normal = 0;
    int target_per_class = max_rows / 2;

    // Leer todo el archivo y balancear 50/50
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;
        std::vector<float> row_features;
        
        for (int i = 0; i < 30; ++i) {
            std::getline(ss, token, ',');
            float val = parse_token(token);
            // Normalización manual básica para evitar explosión de gradientes
            if (i == 0) val = val / 100000.0f; // Time
            if (i == 29) val = val / 1000.0f;  // Amount
            row_features.push_back(val);
        }
        
        std::getline(ss, token, ',');
        float label = parse_token(token);
        
        if (label == 1.0f && count_fraud < target_per_class) {
            all_features.push_back(row_features);
            all_labels.push_back(label);
            count_fraud++;
        } else if (label == 0.0f && count_normal < target_per_class) {
            all_features.push_back(row_features);
            all_labels.push_back(label);
            count_normal++;
        }
        
        if (count_fraud == target_per_class && count_normal == target_per_class) {
            break;
        }
    }
    
    size_t rows = all_features.size();
    if (rows == 0) {
        throw std::runtime_error("El archivo CSV estaba vacío o no contenía datos válidos.");
    }
    
    // Instanciar tensores usando shape {rows, cols}
    utec::algebra::Tensor<float, 2> features(std::array<size_t, 2>{rows, 30});
    utec::algebra::Tensor<float, 2> labels(std::array<size_t, 2>{rows, 1});
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < 30; ++j) {
            features(i, j) = all_features[i][j];
        }
        labels(i, 0) = all_labels[i];
    }
    
    std::cout << "Dataset cargado exitosamente. Filas procesadas: " << rows << "\n";
    return {features, labels};
}

}
