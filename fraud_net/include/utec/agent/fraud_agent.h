#pragma once
#include <memory>
// Ajusta este include si tu NeuralNetwork tiene otro nombre de archivo
#include <utec/nn/neural_network.h>

namespace utec::agent {

class FraudAgent {
private:
    std::unique_ptr<utec::neural_network::NeuralNetwork<float>> model;

public:
    FraudAgent();
    
    // Entrena el modelo usando los features y labels
    void train(const utec::algebra::Tensor<float, 2>& X, const utec::algebra::Tensor<float, 2>& y, size_t epochs, float lr);
    
    // Predice si una transacción es fraude (1) o normal (0)
    int predict(const utec::algebra::Tensor<float, 2>& transaction);
    
    // Evalúa la precisión del modelo en datos de prueba
    void evaluate(const utec::algebra::Tensor<float, 2>& X_test, const utec::algebra::Tensor<float, 2>& y_test);
    
    // Epic 5: Guardar y Cargar Modelo
    void save_model(const std::string& path) const;
    void load_model(const std::string& path);
};

}
