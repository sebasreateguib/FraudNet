#include "utec/agent/fraud_agent.h"
#include "utec/nn/nn_dense.h"
#include "utec/nn/nn_activation.h"
#include "utec/nn/nn_loss.h"
#include "utec/nn/nn_optimizer.h"
#include <iostream>
#include <array>
#include <random>

namespace utec::agent {

FraudAgent::FraudAgent() {
    // Instanciando la red neuronal
    model = std::make_unique<utec::neural_network::NeuralNetwork<float>>();
    
    // Funciones de inicialización de pesos y bias (He initialization simplificada)
    auto init_w = [](utec::algebra::Tensor<float, 2>& t) { 
        std::mt19937 gen(42); // Semilla fija para reproducibilidad
        std::normal_distribution<float> d(0.0f, 0.1f);
        auto shape = t.shape();
        for(size_t i = 0; i < shape[0]; ++i) {
            for(size_t j = 0; j < shape[1]; ++j) {
                t(i, j) = d(gen);
            }
        }
    };
    auto init_b = [](utec::algebra::Tensor<float, 2>& t) { 
        t.fill(0.0f); 
    };

    // 30 variables predictoras -> 16 neuronas
    model->add_layer(std::make_unique<utec::neural_network::Dense<float>>(30, 16, init_w, init_b));
    model->add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    
    // 16 neuronas -> 8 neuronas
    model->add_layer(std::make_unique<utec::neural_network::Dense<float>>(16, 8, init_w, init_b));
    model->add_layer(std::make_unique<utec::neural_network::ReLU<float>>());
    
    // 8 neuronas -> 1 neurona (Salida de clasificación binaria)
    model->add_layer(std::make_unique<utec::neural_network::Dense<float>>(8, 1, init_w, init_b));
    model->add_layer(std::make_unique<utec::neural_network::Sigmoid<float>>());
}

void FraudAgent::train(const utec::algebra::Tensor<float, 2>& X, const utec::algebra::Tensor<float, 2>& y, size_t epochs, float lr) {
    std::cout << "Iniciando entrenamiento del FraudAgent (Detector de Fraude)...\n";
    
    size_t batch_size = 32; // Se le pasa al método, aunque su implementación usa Full Batch por ahora
    
    // Entrenamos usando Binary Cross Entropy Loss y Stochastic Gradient Descent
    model->train<utec::neural_network::BCELoss, utec::neural_network::SGD>(X, y, epochs, batch_size, lr);
    
    std::cout << "Entrenamiento completado.\n";
}

int FraudAgent::predict(const utec::algebra::Tensor<float, 2>& transaction) {
    auto pred = model->predict(transaction);
    std::cerr << "RAW PRED: " << pred(0,0) << std::endl;
    // Si la probabilidad es mayor a 0.5, consideramos que es fraude (1)
    return (pred(0,0) > 0.5f) ? 1 : 0;
}

void FraudAgent::evaluate(const utec::algebra::Tensor<float, 2>& X_test, const utec::algebra::Tensor<float, 2>& y_test) {
    std::cout << "\nEvaluando modelo en datos de prueba...\n";
    
    size_t rows = X_test.shape()[0];
    
    int correct_predictions = 0;
    int false_positives = 0; // Se bloqueó una transacción legítima
    int false_negatives = 0; // Se dejó pasar un fraude
    int true_positives = 0;  // Fraude atrapado
    int true_negatives = 0;  // Transacción legítima aprobada
    
    utec::algebra::Tensor<float, 2> row_tensor(std::array<size_t, 2>{1, 30});
    
    for (size_t i = 0; i < rows; ++i) {
        for(size_t j = 0; j < 30; ++j) {
            row_tensor(0, j) = X_test(i, j);
        }
        
        int pred = predict(row_tensor);
        int actual = static_cast<int>(y_test(i, 0));
        
        if (pred == actual) correct_predictions++;
        
        if (pred == 1 && actual == 0) false_positives++;
        if (pred == 0 && actual == 1) false_negatives++;
        if (pred == 1 && actual == 1) true_positives++;
        if (pred == 0 && actual == 0) true_negatives++;
    }
    
    float accuracy = (static_cast<float>(correct_predictions) / rows) * 100.0f;
    
    std::cout << "=== Resultados de Evaluación ===\n";
    std::cout << "Exactitud (Accuracy): " << accuracy << "%\n";
    std::cout << "Verdaderos Fraudes Atrapados (TP): " << true_positives << "\n";
    std::cout << "Falsos Positivos (Tarjetas sanas bloqueadas): " << false_positives << "\n";
    std::cout << "Falsos Negativos (Fraudes que escaparon): " << false_negatives << "\n";
}

void FraudAgent::save_model(const std::string& path) const {
    model->save(path);
}

void FraudAgent::load_model(const std::string& path) {
    model->load(path);
}

}
